import base64
import getpass
import inspect
import logging
import os
import shutil
import subprocess
import sys
import urllib2
from datetime import datetime

import h5py
import keyring
import numpy as np
import pycurl
import pyrttov
from osgeo import gdal, osr
from pydap import client
from pydap.cas import urs
from pyproj import Proj

LOGGER = logging.getLogger('pydisalexi.geotiff')


def folders(base):
    dataBase = os.path.join(base, 'data')
    landsatDataBase = os.path.join(dataBase, 'Landsat-8')
    asterDataBase = os.path.join(dataBase, 'ASTER')
    landsat_SR = os.path.join(landsatDataBase, 'SR')
    if not os.path.exists(landsat_SR):
        os.makedirs(landsat_SR)
    landsat_Temp = os.path.join(landsat_SR, 'temp')
    if not os.path.exists(landsat_Temp):
        os.makedirs(landsat_Temp)
    landsat_LST = os.path.join(landsatDataBase, 'LST')
    if not os.path.exists(landsat_LST):
        os.makedirs(landsat_LST)
    asterEmissivityBase = os.path.join(asterDataBase, 'asterEmissivity')
    if not os.path.exists(asterEmissivityBase):
        os.makedirs(asterEmissivityBase)
    landsatEmissivityBase = os.path.join(asterDataBase, 'landsatEmissivity')
    if not os.path.exists(landsatEmissivityBase):
        os.makedirs(landsatEmissivityBase)
    ASTERmosaicTemp = os.path.join(asterDataBase, 'mosaicTemp')
    if not os.path.exists(ASTERmosaicTemp):
        os.makedirs(ASTERmosaicTemp)
    out = {'landsat_LST': landsat_LST, 'landsat_SR': landsat_SR,
           'asterEmissivityBase': asterEmissivityBase, 'ASTERmosaicTemp': ASTERmosaicTemp,
           'landsatDataBase': landsatDataBase, 'landsatEmissivityBase': landsatEmissivityBase,
           'landsat_Temp': landsat_Temp}
    return out


class earthDataHTTPRedirectHandler(urllib2.HTTPRedirectHandler):
    def http_error_302(self, req, fp, code, msg, headers):
        return urllib2.HTTPRedirectHandler.http_error_302(self, req, fp, code, msg, headers)


def getHTTPdata(url, outFN, auth=None):
    request = urllib2.Request(url)
    if not (auth == None):
        username = auth[0]
        password = auth[1]
        base64string = base64.encodestring('%s:%s' % (username, password)).replace('\n', '')
        request.add_header("Authorization", "Basic %s" % base64string)

    cookieprocessor = urllib2.HTTPCookieProcessor()
    opener = urllib2.build_opener(earthDataHTTPRedirectHandler, cookieprocessor)
    urllib2.install_opener(opener)
    r = opener.open(request)
    result = r.read()

    with open(outFN, 'wb') as f:
        f.write(result)


def writeArray2Tiff(data, lats, lons, outfile):
    Projection = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    xres = lons[1] - lons[0]
    yres = lats[1] - lats[0]

    ysize = len(lats)
    xsize = len(lons)

    ulx = lons[0]  # - (xres / 2.)
    uly = lats[0]  # - (yres / 2.)
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(outfile, xsize, ysize, 1, gdal.GDT_Float32)

    srs = osr.SpatialReference()
    if isinstance(Projection, basestring):
        srs.ImportFromProj4(Projection)
    else:
        srs.ImportFromEPSG(Projection)
    ds.SetProjection(srs.ExportToWkt())

    gt = [ulx, xres, 0, uly, 0, yres]
    ds.SetGeoTransform(gt)

    outband = ds.GetRasterBand(1)
    outband.WriteArray(data)
    ds.FlushCache()

    ds = None


def _test_outside(testx, lower, upper):
    """
    True if testx, or any element of it is outside [lower, upper].

    Both lower bound and upper bound included
    Input: Integer or floating point scalar or Numpy array.
    """
    test = np.array(testx)
    return np.any(test < lower) or np.any(test > upper)


# custom exception
class RasterError(Exception):
    """Custom exception for errors during raster processing in Pygaarst"""
    pass


class landsat_metadata:
    """
    A landsat metadata object. This class builds is attributes
    from the names of each tag in the xml formatted .MTL files that
    come with landsat data. So, any tag that appears in the MTL file
    will populate as an attribute of landsat_metadata.

    You can access explore these attributes by using, for example

    .. code-block:: python

        from dnppy import landsat
        meta = landsat.landsat_metadata(my_filepath) # create object

        from pprint import pprint                    # import pprint
        pprint(vars(m))                              # pretty print output
        scene_id = meta.LANDSAT_SCENE_ID             # access specific attribute

    :param filename: the filepath to an MTL file.
    """

    def __init__(self, filename):
        """
        There are several critical attributes that keep a common
        naming convention between all landsat versions, so they are
        initialized in this class for good record keeping and reference
        """

        # custom attribute additions
        self.FILEPATH = filename
        self.DATETIME_OBJ = None

        # product metadata attributes
        self.LANDSAT_SCENE_ID = None
        self.DATA_TYPE = None
        self.ELEVATION_SOURCE = None
        self.OUTPUT_FORMAT = None
        self.SPACECRAFT_ID = None
        self.SENSOR_ID = None
        self.WRS_PATH = None
        self.WRS_ROW = None
        self.NADIR_OFFNADIR = None
        self.TARGET_WRS_PATH = None
        self.TARGET_WRS_ROW = None
        self.DATE_ACQUIRED = None
        self.SCENE_CENTER_TIME = None

        # image attributes
        self.CLOUD_COVER = None
        self.IMAGE_QUALITY_OLI = None
        self.IMAGE_QUALITY_TIRS = None
        self.ROLL_ANGLE = None
        self.SUN_AZIMUTH = None
        self.SUN_ELEVATION = None
        self.EARTH_SUN_DISTANCE = None  # calculated for Landsats before 8.

        # read the file and populate the MTL attributes
        self._read(filename)

    def _read(self, filename):
        """ reads the contents of an MTL file """

        # if the "filename" input is actually already a metadata class object, return it back.
        if inspect.isclass(filename):
            return filename

        fields = []
        values = []

        metafile = open(filename, 'r')
        metadata = metafile.readlines()

        for line in metadata:
            # skips lines that contain "bad flags" denoting useless data AND lines
            # greater than 1000 characters. 1000 character limit works around an odd LC5
            # issue where the metadata has 40,000+ characters of whitespace
            bad_flags = ["END", "GROUP"]
            if not any(x in line for x in bad_flags) and len(line) <= 1000:
                try:
                    line = line.replace("  ", "")
                    line = line.replace("\n", "")
                    field_name, field_value = line.split(' = ')
                    fields.append(field_name)
                    values.append(field_value)
                except:
                    pass

        for i in range(len(fields)):

            # format fields without quotes,dates, or times in them as floats
            if not any(['"' in values[i], 'DATE' in fields[i], 'TIME' in fields[i]]):
                setattr(self, fields[i], float(values[i]))
            else:
                values[i] = values[i].replace('"', '')
                setattr(self, fields[i], values[i])

        # create datetime_obj attribute (drop decimal seconds)
        dto_string = self.DATE_ACQUIRED + self.SCENE_CENTER_TIME
        self.DATETIME_OBJ = datetime.strptime(dto_string.split(".")[0], "%Y-%m-%d%H:%M:%S")
        print("Scene {0} center time is {1}".format(self.LANDSAT_SCENE_ID, self.DATETIME_OBJ))


class GeoTIFF(object):
    """
    Represents a GeoTIFF file for data access and processing and provides
    a number of useful methods and attributes.

    Arguments:
      filepath (str): the full or relative file path
    """

    def __init__(self, filepath):
        try:
            self.dataobj = gdal.Open(filepath)
        except RuntimeError as err:
            LOGGER.error("Could not open %s: %s" % (filepath, err.message))
            raise
        self.filepath = filepath
        self.ncol = self.dataobj.RasterXSize
        self.nrow = self.dataobj.RasterYSize
        self.nbands = self.dataobj.RasterCount
        self._gtr = self.dataobj.GetGeoTransform()
        # see http://www.gdal.org/gdal_datamodel.html
        self.ulx = self._gtr[0]
        self.uly = self._gtr[3]
        self.lrx = (self.ulx + self.ncol * self._gtr[1]
                    + self.nrow * self._gtr[2])
        self.lry = (self.uly + self.ncol * self._gtr[4]
                    + self.nrow * self._gtr[5])
        if self._gtr[2] != 0 or self._gtr[4] != 0:
            LOGGER.warning(
                "The dataset is not north-up. The geotransform is given "
                + "by: (%s). " % ', '.join([str(item) for item in self._gtr])
                + "Northing and easting values will not have expected meaning."
            )
        self.dataobj = None

    @property
    def data(self):
        """2D numpy array for single-band GeoTIFF file data. Otherwise, 3D. """
        if not self.dataobj:
            self.dataobj = gdal.Open(self.filepath)
        dat = self.dataobj.ReadAsArray()
        self.dataobj = None
        return dat

    @property
    def projection(self):
        """The dataset's coordinate reference system as a Well-Known String"""
        if not self.dataobj:
            self.dataobj = gdal.Open(self.filepath)
        dat = self.dataobj.GetProjection()
        self.dataobj = None
        return dat

    @property
    def proj4(self):
        """The dataset's coordinate reference system as a PROJ4 string"""
        osrref = osr.SpatialReference()
        osrref.ImportFromWkt(self.projection)
        return osrref.ExportToProj4()

    @property
    def coordtrans(self):
        """A PROJ4 Proj object, which is able to perform coordinate
        transformations"""
        return Proj(self.proj4)

    @property
    def delx(self):
        """The sampling distance in x-direction, in physical units
        (eg metres)"""
        return self._gtr[1]

    @property
    def dely(self):
        """The sampling distance in y-direction, in physical units
        (eg metres). Negative in northern hemisphere."""
        return self._gtr[5]

    @property
    def easting(self):
        """The x-coordinates of first row pixel corners,
        as a numpy array: upper-left corner of upper-left pixel
        to upper-right corner of upper-right pixel (ncol+1)."""
        delta = np.abs(
            (self.lrx - self.ulx) / self.ncol
            - self.delx
        )
        if delta > 10e-2:
            LOGGER.warn(
                "GeoTIFF issue: E-W grid step differs from "
                + "deltaX by more than 1% ")
        return np.linspace(self.ulx, self.lrx, self.ncol + 1)

    @property
    def northing(self):
        """The y-coordinates of first column pixel corners,
        as a numpy array: lower-left corner of lower-left pixel to
        upper-left corner of upper-left pixel (nrow+1)."""
        # check if data grid step is consistent
        delta = np.abs(
            (self.lry - self.uly) / self.nrow
            - self.dely
        )
        if delta > 10e-2:
            LOGGER.warn(
                "GeoTIFF issue: N-S grid step differs from "
                + "deltaY by more than 1% ")
        return np.linspace(self.lry, self.uly, self.nrow + 1)

    @property
    def x_pxcenter(self):
        """The x-coordinates of pixel centers, as a numpy array ncol."""
        return np.linspace(
            self.ulx + self.delx / 2,
            self.lrx - self.delx / 2,
            self.ncol)

    @property
    def y_pxcenter(self):
        """y-coordinates of pixel centers, nrow."""
        return np.linspace(
            self.lry - self.dely / 2,
            self.uly + self.dely / 2,
            self.nrow)

    @property
    def _XY(self):
        """Meshgrid of nrow+1, ncol+1 corner xy coordinates"""
        return np.meshgrid(self.easting, self.northing)

    @property
    def _XY_pxcenter(self):
        """Meshgrid of nrow, ncol center xy coordinates"""
        return np.meshgrid(self.x_pxcenter, self.y_pxcenter)

    @property
    def _LonLat_pxcorner(self):
        """Meshgrid of nrow+1, ncol+1 corner Lon/Lat coordinates"""
        return self.coordtrans(*self._XY, inverse=True)

    @property
    def _LonLat_pxcenter(self):
        """Meshgrid of nrow, ncol center Lon/Lat coordinates"""
        return self.coordtrans(*self._XY_pxcenter, inverse=True)

    @property
    def Lon(self):
        """Longitude coordinate of each pixel corner, as an array"""
        return self._LonLat_pxcorner[0]

    @property
    def Lat(self):
        """Latitude coordinate of each pixel corner, as an array"""
        return self._LonLat_pxcorner[1]

    @property
    def Lon_pxcenter(self):
        """Longitude coordinate of each pixel center, as an array"""
        return self._LonLat_pxcenter[0]

    @property
    def Lat_pxcenter(self):
        """Latitude coordinate of each pixel center, as an array"""
        return self._LonLat_pxcenter[1]

    def ij2xy(self, i, j):
        """
        Converts array index pair(s) to easting/northing coordinate pairs(s).

        NOTE: array coordinate origin is in the top left corner whereas
        easting/northing origin is in the bottom left corner. Easting and
        northing are floating point numbers, and refer to the top-left corner
        coordinate of the pixel. i runs from 0 to nrow-1, j from 0 to ncol-1.
        For i=nrow and j=ncol, the bottom-right corner coordinate of the
        bottom-right pixel will be returned. This is identical to the bottom-
        right corner.

        Arguments:
            i (int): scalar or array of row coordinate index
            j (int): scalar or array of column coordinate index

        Returns:
            x (float): scalar or array of easting coordinates
            y (float): scalar or array of northing coordinates
        """
        if (_test_outside(i, 0, self.nrow)
                or _test_outside(j, 0, self.ncol)):
            raise RasterError(
                "Coordinates %d, %d out of bounds" % (i, j))
        x = self.easting[0] + j * self.delx
        y = self.northing[-1] + i * self.dely
        return x, y

    def xy2ij(self, x, y, precise=False):
        """
        Convert easting/northing coordinate pair(s) to array coordinate
        pairs(s).

        NOTE: see note at ij2xy()

        Arguments:
            x (float): scalar or array of easting coordinates
            y (float): scalar or array of northing coordinates
            precise (bool): if true, return fractional array coordinates

        Returns:
            i (int, or float): scalar or array of row coordinate index
            j (int, or float): scalar or array of column coordinate index
        """
        if (_test_outside(x, self.easting[0], self.easting[-1]) or
                _test_outside(y, self.northing[0], self.northing[-1])):
            raise RasterError("Coordinates out of bounds")
        i = (1 - (y - self.northing[0]) /
             (self.northing[-1] - self.northing[0])) * self.nrow
        j = ((x - self.easting[0]) /
             (self.easting[-1] - self.easting[0])) * self.ncol
        if precise:
            return i, j
        else:
            return int(np.floor(i)), int(np.floor(j))

    def clone(self, newpath, newdata):
        """
        Creates new GeoTIFF object from existing: new data, same georeference.

        Arguments:
            newpath: valid file path
            newdata: numpy array, 2 or 3-dim

        Returns:
            A raster.GeoTIFF object
        """
        # convert Numpy dtype objects to GDAL type codes
        # see https://gist.github.com/chryss/8366492

        NPDTYPE2GDALTYPECODE = {
            "uint8": 1,
            "int8": 1,
            "uint16": 2,
            "int16": 3,
            "uint32": 4,
            "int32": 5,
            "float32": 6,
            "float64": 7,
            "complex64": 10,
            "complex128": 11,
        }
        # check if newpath is potentially a valid file path to save data
        dirname, fname = os.path.split(newpath)
        if dirname:
            if not os.path.isdir(dirname):
                print("%s is not a valid directory to save file to " % dirname)
        if os.path.isdir(newpath):
            LOGGER.warning(
                "%s is a directory." % dirname + " Choose a name "
                + "that is suitable for writing a dataset to.")
        if (newdata.shape != self.data.shape
                and newdata.shape != self.data[0, ...].shape):
            raise RasterError(
                "New and cloned GeoTIFF dataset must be the same shape.")
        dims = newdata.ndim
        if dims == 2:
            bands = 1
        elif dims > 2:
            bands = newdata.shape[0]
        else:
            raise RasterError(
                "New data array has only %s dimensions." % dims)
        try:
            LOGGER.info(newdata.dtype.name)
            LOGGER.info(NPDTYPE2GDALTYPECODE)
            LOGGER.info(NPDTYPE2GDALTYPECODE[newdata.dtype.name])
            gdaltype = NPDTYPE2GDALTYPECODE[newdata.dtype.name]
        except KeyError as err:
            raise RasterError(
                "Data type in array %s " % newdata.dtype.name
                + "cannot be converted to GDAL data type: \n%s" % err.message)
        proj = self.projection
        geotrans = self._gtr
        gtiffdr = gdal.GetDriverByName('GTiff')
        gtiff = gtiffdr.Create(newpath, self.ncol, self.nrow, bands, gdaltype)
        gtiff.SetProjection(proj)
        gtiff.SetGeoTransform(geotrans)
        if dims == 2:
            gtiff.GetRasterBand(1).WriteArray(newdata)
        else:
            for idx in range(dims):
                gtiff.GetRasterBand(idx + 1).WriteArray(newdata[idx, :, :])
        gtiff = None
        return GeoTIFF(newpath)


class rttov:
    def __init__(self, username, password, ulLat, ulLon, lrLat, lrLon, sza, saa, d):
        # base = os.getcwd()
        # folder_paths = folders(base)
        self.earthLoginUser = username
        self.earthLoginPass = password
        # self.landsatSR = folder_paths['landsat_SR']
        # meta = landsat_metadata(filepath)
        # self.productID = meta.LANDSAT_PRODUCT_ID
        # self.scene = self.productID.split('_')[2]
        self.ulLat = ulLat
        self.ulLon = ulLon
        self.lrLat = lrLat
        self.lrLon = lrLon
        self.solZen = sza
        self.solAzi = saa
        # d = meta.DATETIME_OBJ
        self.year = d.year
        self.month = d.month
        self.day = d.day
        self.hr = d.hour  # UTC

    def prepare_profile_data(self):

        ul = [self.ulLon - 1.5, self.ulLat + 1.5]
        lr = [self.lrLon + 1.5, self.lrLat - 1.5]
        # The data is lat/lon and upside down so [0,0] = [-90.0,-180.0]
        max_x = int((lr[0] - (-180)) / 0.625)
        min_x = int((ul[0] - (-180)) / 0.625)
        min_y = int((lr[1] - (-90)) / 0.5)
        max_y = int((ul[1] - (-90)) / 0.5)

        if self.year < 1992:
            file_type = 100
        elif 1991 < self.year < 2001:
            file_type = 200
        elif 2000 < self.year < 2011:
            file_type = 300
        else:
            file_type = 400

        # Instantaneous Two-Dimensional Collections
        # inst1_2d_asm_Nx (M2I1NXASM): Single-Level Diagnostics
        # =============================================================================
        opendap_url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/'
        product = 'M2I1NXASM.5.12.4'
        filename = 'MERRA2_%d.inst1_2d_asm_Nx.%04d%02d%02d.nc4' % (file_type, self.year, self.month, self.day)
        full_url = os.path.join(opendap_url, product, '%04d' % self.year, '%02d' % self.month, filename)
        session = urs.setup_session(username=self.earthLoginUser,
                                    password=self.earthLoginPass,
                                    check_url=full_url)
        d = client.open_url(full_url, session=session)
        # surface presure [Pa]
        surface_pressure = d.PS
        sp = np.squeeze(surface_pressure[self.hr, min_y:max_y, min_x:max_x] / 100.)  # Pa to kPa
        sprshp = np.reshape(sp, sp.shape[0] * sp.shape[1])

        # 2m air Temp (K)
        temp_2m = d.T2M
        t2 = np.squeeze(temp_2m[self.hr, min_y:max_y, min_x:max_x])
        t2rshp = np.reshape(t2, t2.shape[0] * t2.shape[1])

        # 2m specific humidity [kg kg -1] -> 2 m water vapor [ppmv]
        spc_hum = d.QV2M
        # wv_mmr = 1.e-6 * wv_ppmv_layer * (Rair / Rwater)
        # wv_mmr in kg/kg, Rair = 287.0, Rwater = 461.5
        q = np.squeeze(spc_hum[self.hr, min_y:max_y, min_x:max_x])
        q2 = q / (1e-6 * (287.0 / 461.5))
        q2_reshape = np.reshape(q2, q2.shape[0] * q2.shape[1])

        # skin temp [K]
        sktIn = d.TS
        skt = np.squeeze(sktIn[self.hr, min_y:max_y, min_x:max_x])
        sktrshp = np.reshape(skt, skt.shape[0] * skt.shape[1])

        # U10M 10-meter_eastward_wind [m s-1]
        u10In = d.U10M
        u10 = np.squeeze(u10In[self.hr, min_y:max_y, min_x:max_x])
        u10rshp = np.reshape(u10, u10.shape[0] * u10.shape[1])

        # V10M 10-meter_northward_wind [m s-1]
        v10In = d.V10M
        v10 = np.squeeze(v10In[self.hr, min_y:max_y, min_x:max_x])
        v10rshp = np.reshape(v10, v10.shape[0] * v10.shape[1])

        opendap_url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/'
        product = 'M2I3NVASM.5.12.4'
        filename = 'MERRA2_%d.inst3_3d_asm_Nv.%04d%02d%02d.nc4' % (file_type, self.year, self.month, self.day)
        full_url = os.path.join(opendap_url, product, '%04d' % self.year, '%02d' % self.month, filename)
        session = urs.setup_session(username=self.earthLoginUser,
                                    password=self.earthLoginPass,
                                    check_url=full_url)
        d = client.open_url(full_url, session=session)
        hr = int(np.round(self.hr / 3.))  # convert from 1 hr to 3 hr dataset

        # layers specific humidity [kg kg -1] -> 2 m water vapor [ppmv]
        qvIn = d.QV
        # wv_mmr = 1.e-6 * wv_ppmv_layer * (Rair / Rwater)
        # wv_mmr in kg/kg, Rair = 287.0, Rwater = 461.5
        qv = np.squeeze(qvIn[hr, :, min_y:max_y, min_x:max_x])
        qv = qv / (1e-6 * (287.0 / 461.5))
        qvrshp = np.reshape(qv, [qv.shape[0], qv.shape[1] * qv.shape[2]]).T

        # layers air temperature [K]
        tIn = d.T
        # wv_mmr = 1.e-6 * wv_ppmv_layer * (Rair / Rwater)
        # wv_mmr in kg/kg, Rair = 287.0, Rwater = 461.5
        t = np.squeeze(tIn[hr, :, min_y:max_y, min_x:max_x])
        trshp = np.reshape(t, [t.shape[0], t.shape[1] * t.shape[2]]).T

        # mid_level_pressure [Pa]

        plIn = d.PL
        pl = np.squeeze(plIn[hr, :, min_y:max_y, min_x:max_x] / 100)  # Pa to kPa
        plrshp = np.reshape(pl, [pl.shape[0], pl.shape[1] * pl.shape[2]]).T

        LAT = d.lat
        LON = d.lon
        lats = LAT[:]
        lons = LON[:]
        lat = np.tile(lats, (len(lons), 1)).T
        latIn = np.squeeze(lat[min_y:max_y, min_x:max_x])
        latrshp = np.reshape(latIn, latIn.shape[0] * latIn.shape[1])
        lon = np.tile(lons, (len(lats), 1))
        lonIn = np.squeeze(lon[min_y:max_y, min_x:max_x])
        lonrshp = np.reshape(lonIn, lonIn.shape[0] * lonIn.shape[1])
        el = np.repeat(0.0, v10.shape[0] * v10.shape[1])  # NEED DEM
        # check surface pressure

        sunzen = np.repeat(self.solZen, v10.shape[0] * v10.shape[1])
        sunazi = np.repeat(self.solAzi, v10.shape[0] * v10.shape[1])
        fetch = np.repeat(100000, v10.shape[0] * v10.shape[1])
        satzen = np.repeat(0.0, v10.shape[0] * v10.shape[1])
        satazi = np.repeat(0.0, v10.shape[0] * v10.shape[1])

        # Units for gas profiles
        gas_units = 2  # ppmv over moist air

        # datetimes[6][nprofiles]: yy, mm, dd, hh, mm, ss
        datetimes = np.tile([self.year, self.month, self.day, hr, 0, 0], (v10.shape[0] * v10.shape[1], 1))

        # angles[4][nprofiles]: satzen, satazi, sunzen, sunazi
        # get from landsat MTL
        angles = np.vstack((satzen, satazi, sunzen, sunazi)).T

        # surftype[2][nprofiles]: surftype, watertype
        surftype = np.zeros([angles.shape[0], 2])  # NEED LAND/WATER mask

        # surfgeom[3][nprofiles]: lat, lon, elev
        surfgeom = np.vstack((latrshp, lonrshp, el)).T

        # s2m[6][nprofiles]: 2m p, 2m t, 2m q, 10m wind u, v, wind fetch
        s2m = np.vstack((sprshp, t2rshp, q2_reshape, u10rshp, v10rshp, fetch)).T

        # skin[9][nprofiles]: skin T, salinity, snow_frac, foam_frac, fastem_coefsx5
        sal = np.repeat(35.0, v10.shape[0] * v10.shape[1])
        snow_frac = np.repeat(0.0, v10.shape[0] * v10.shape[1])
        foam_frac = np.repeat(0.0, v10.shape[0] * v10.shape[1])
        fastem_coef1 = np.repeat(3.0, v10.shape[0] * v10.shape[1])
        fastem_coef2 = np.repeat(5.0, v10.shape[0] * v10.shape[1])
        fastem_coef3 = np.repeat(15.0, v10.shape[0] * v10.shape[1])
        fastem_coef4 = np.repeat(0.1, v10.shape[0] * v10.shape[1])
        fastem_coef5 = np.repeat(0.3, v10.shape[0] * v10.shape[1])

        skin = np.vstack((sktrshp, sal, snow_frac, foam_frac, fastem_coef1, fastem_coef2, fastem_coef3, fastem_coef4,
                          fastem_coef5)).T

        outDict = {'P': plrshp, 'T': trshp, 'Q': qvrshp, 'Angles': angles, 'S2m': s2m, \
                   'Skin': skin, 'SurfType': surftype, 'SurfGeom': surfgeom, 'Datetimes': datetimes, \
                   'origShape': t2.shape}

        return outDict


class Landsat:
    def __init__(self, filepath, username, password):
        base = os.getcwd()
        Folders = folders(base)
        self.earthLoginUser = username
        self.earthLoginPass = password
        self.landsatLST = Folders['landsat_LST']
        self.landsatSR = Folders['landsat_SR']
        self.landsatTemp = Folders['landsat_Temp']
        self.asterEmissivityBase = Folders['asterEmissivityBase']
        self.ASTERmosaicTemp = Folders['ASTERmosaicTemp']
        self.landsatDataBase = Folders['landsatDataBase']
        self.landsatEmissivityBase = Folders['landsatEmissivityBase']
        self.filepath = filepath
        meta = landsat_metadata(filepath)
        self.productID = meta.LANDSAT_PRODUCT_ID
        self.sceneID = meta.LANDSAT_SCENE_ID
        self.scene = self.productID.split('_')[2]
        self.ls = GeoTIFF(os.path.join(os.sep.join(filepath.split(os.sep)[:-1]), '%s_sr_band1.tif' % self.productID))
        self.proj4 = self.ls.proj4
        self.inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        self.ulx = meta.CORNER_UL_PROJECTION_X_PRODUCT
        self.uly = meta.CORNER_UL_PROJECTION_Y_PRODUCT
        self.lrx = meta.CORNER_LR_PROJECTION_X_PRODUCT
        self.lry = meta.CORNER_LR_PROJECTION_Y_PRODUCT
        self.ulLat = meta.CORNER_UL_LAT_PRODUCT
        self.ulLon = meta.CORNER_UL_LON_PRODUCT
        self.lrLat = meta.CORNER_LR_LAT_PRODUCT
        self.lrLon = meta.CORNER_LR_LON_PRODUCT
        self.delx = meta.GRID_CELL_SIZE_REFLECTIVE
        self.dely = meta.GRID_CELL_SIZE_REFLECTIVE
        self.solZen = meta.SUN_ELEVATION
        self.solAzi = meta.SUN_AZIMUTH
        self.landsatDate = meta.DATE_ACQUIRED
        self.landsatTime = meta.SCENE_CENTER_TIME[:-2]
        self.Kappa1 = meta.K1_CONSTANT_BAND_10
        self.Kappa2 = meta.K2_CONSTANT_BAND_10
        d = datetime.strptime('%s%s' % (self.landsatDate, self.landsatTime), '%Y-%m-%d%H:%M:%S.%f')
        self.year = d.year
        self.month = d.month
        self.day = d.day
        self.hr = d.hour  # UTC

    def processASTERemis(self):
        ASTERurlBase = 'https://e4ftl01.cr.usgs.gov/ASTT/AG100.003/2000.01.01'
        # use Landsat scene area
        latRange = range(int(np.floor(self.lrLat)), int(np.ceil(self.ulLat)) + 1)
        lonRange = range(int(np.floor(self.ulLon)), int(np.ceil(self.lrLon)) + 1)
        for i in range(len(latRange)):
            for j in range(len(lonRange)):
                latString = "{1:0{0}d}".format(2 if latRange[i] >= 0 else 3, latRange[i])
                lonString = "{1:0{0}d}".format(3 if lonRange[j] >= 0 else 4, lonRange[j])
                asterFN = 'AG100.v003.%s.%s.0001.h5' % (latString, lonString)
                # ASTER Emissivity product AG100 comes in 1 x 1 degree tiles where UL is given in the filename.
                ASTERurl = os.path.join(ASTERurlBase, asterFN)
                # print ASTERurl
                localAsterFN = os.path.join(self.asterEmissivityBase, asterFN)

                if not os.path.isfile(localAsterFN):
                    print "downloading ASTER..."
                    try:
                        getHTTPdata(ASTERurl, localAsterFN, (self.earthLoginUser, self.earthLoginPass))
                    except Exception:
                        print("failed to get the file")
                        continue
                        # open HDF file, extract the desired dataset and save to GTiff

                fh5 = h5py.File(localAsterFN, 'r')
                EmisBand4 = np.array(fh5["/Emissivity/Mean/"])[3] / 1000.
                lats = np.array(fh5["/Geolocation/Latitude/"])
                lons = np.array(fh5["/Geolocation/Longitude/"])
                tempName = os.path.join(self.ASTERmosaicTemp, 'emis%s%s.tiff' % (latString, lonString))
                writeArray2Tiff(EmisBand4, lats[:, 0], lons[0, :], tempName)

        # mosaic ,reproject and save as geotiff
        mosaicTempFN = '%s/mosaic.vrt' % self.ASTERmosaicTemp
        mosaicVRTcommand = 'gdalbuildvrt -srcnodata 0 %s %s/*.tiff' % (mosaicTempFN, self.ASTERmosaicTemp)
        out = subprocess.check_output(mosaicVRTcommand, shell=True)
        resampName = os.path.join(self.landsatEmissivityBase, '%s_EMIS.tiff' % self.sceneID)
        command = "gdalwarp -overwrite -s_srs '%s' -t_srs '%s' -r bilinear -tr 30 30 -te %f %f %f %f -of GTiff %s %s" % (
            self.inProj4, self.proj4, self.ulx, self.lry, self.lrx, self.uly, mosaicTempFN, resampName)
        out = subprocess.check_output(command, shell=True)
        print(out)
        print 'done processing ASTER'
        shutil.rmtree(self.ASTERmosaicTemp)
        os.makedirs(self.ASTERmosaicTemp)
        return resampName

    def processLandsatLST(self, tirsRttov, merraDict):

        # Landsat brightness temperature
        landsat = os.path.join(os.sep.join(self.filepath.split(os.sep)[:-1]), "%s_bt_band10.tif" % self.productID)
        Lg = gdal.Open(landsat)
        BT = Lg.ReadAsArray() / 10.
        # =====get radiance from BT=========
        ThermalRad = self.Kappa1 / (np.exp(self.Kappa2 / BT) - 1)
        Lg = None

        origShap = merraDict['origShape']
        surfgeom = merraDict['SurfGeom']
        nlevels = merraDict['P'].shape[1]

        # reshape and resize image to fit landsat
        lats = np.flipud(np.resize(surfgeom[:, 0], origShap))
        lons = np.flipud(np.resize(surfgeom[:, 1], origShap))

        channel = 1

        nu4 = 1 / (10.895 * 0.0001)  # convert to cm

        # Process downwelling radiance
        RadDown = np.flipud(np.resize(tirsRttov.Rad2Down[:, channel, nlevels - 2], origShap))
        tempName = os.path.join(self.landsatDataBase, 'RadDown.tiff')
        resampName = os.path.join('%sReproj.tiff' % tempName[:-4])
        writeArray2Tiff(RadDown, lats[:, 0], lons[0, :], tempName)

        command = "gdalwarp -overwrite -s_srs '%s' -t_srs '%s' -r bilinear -tr 30 30 -te %d %d %d %d -of GTiff %s %s" % (
            self.inProj4, self.proj4, self.ulx, self.lry, self.lrx, self.uly, tempName, resampName)
        out = subprocess.check_output(command, shell=True)
        resampName2 = resampName[:-4] + '2.tiff'
        command = "gdalwarp -overwrite -ts %d %d -of GTiff %s %s" % (
            ThermalRad.shape[1], ThermalRad.shape[0], resampName, resampName2)
        out = subprocess.check_output(command, shell=True)
        Lg = gdal.Open(resampName2)
        RadDown = Lg.ReadAsArray()
        RadDown = (RadDown * (nu4 ** 2 / 10 ** 7))  # *.001
        Lg = None

        # Process upwelling radiance
        RadUp = np.flipud(np.resize(tirsRttov.Rad2Up[:, channel, nlevels - 2], origShap))
        tempName = os.path.join(self.landsatDataBase, 'RadUp.tiff')
        resampName = os.path.join('%sReproj.tiff' % tempName[:-4])

        writeArray2Tiff(RadUp, lats[:, 0], lons[0, :], tempName)
        command = "gdalwarp -overwrite -s_srs '%s' -t_srs '%s' -r bilinear -tr 30 30 -te %d %d %d %d -of GTiff %s %s" % (
            self.inProj4, self.proj4, self.ulx, self.lry, self.lrx, self.uly, tempName, resampName)
        out = subprocess.check_output(command, shell=True)
        resampName2 = resampName[:-4] + '2.tiff'
        command = "gdalwarp -overwrite -ts %d %d -of GTiff %s %s" % (
            ThermalRad.shape[1], ThermalRad.shape[0], resampName, resampName2)
        out = subprocess.check_output(command, shell=True)
        Lg = gdal.Open(resampName2)
        RadUp = Lg.ReadAsArray()
        RadUp = (RadUp * (nu4 ** 2 / 10 ** 7))  # *.001
        Lg = None

        # Process transmission
        trans = np.flipud(np.resize(tirsRttov.TauTotal[:, channel], origShap))
        tempName = os.path.join(self.landsatDataBase, 'trans.tiff')
        resampName = os.path.join('%sReproj.tiff' % tempName[:-4])
        writeArray2Tiff(trans, lats[:, 0], lons[0, :], tempName)

        command = "gdalwarp -overwrite -s_srs '%s' -t_srs '%s' -r bilinear -tr 30 30 -te %d %d %d %d -of GTiff %s %s" % (
            self.inProj4, self.proj4, self.ulx, self.lry, self.lrx, self.uly, tempName, resampName)
        out = subprocess.check_output(command, shell=True)
        resampName2 = resampName[:-4] + '2.tiff'
        command = "gdalwarp -overwrite -ts %d %d -of GTiff %s %s" % (
            ThermalRad.shape[1], ThermalRad.shape[0], resampName, resampName2)
        out = subprocess.check_output(command, shell=True)
        Lg = gdal.Open(resampName2)
        trans = Lg.ReadAsArray()
        Lg = None

        # get emissivity from ASTER

        if not os.path.exists(os.path.join(self.landsatEmissivityBase, '%s_EMIS.tiff' % self.sceneID)):
            ASTERemisFNtemp = self.processASTERemis()
            ASTERemisFN = ASTERemisFNtemp[:-4] + '2.tiff'
            command = "gdalwarp -overwrite -ts %d %d -of GTiff %s %s" % (
                trans.shape[1], trans.shape[0], ASTERemisFNtemp, ASTERemisFN)
            out = subprocess.check_output(command, shell=True)
        else:
            ASTERemisFNtemp = os.path.join(self.landsatEmissivityBase, '%s_EMIS.tiff' % self.sceneID)
            ASTERemisFN = ASTERemisFNtemp[:-4] + '2.tiff'
            command = "gdalwarp -overwrite -ts %d %d -of GTiff %s %s" % (
                trans.shape[1], trans.shape[0], ASTERemisFNtemp, ASTERemisFN)
            out = subprocess.check_output(command, shell=True)

        aster = gdal.Open(ASTERemisFN)
        emis = aster.ReadAsArray()
        aster = None
        # calcualte LST
        emis[emis < 0.000001] = np.nan
        surfRad = (((ThermalRad - RadUp) / trans) - (1 - emis) * RadDown) / emis
        # get Kappa constants from Landsat

        LST = np.array(self.Kappa2 * (1 / np.log(self.Kappa1 / surfRad)), dtype='float32')

        lstName = os.path.join(self.landsatTemp, '%s_lst.tiff' % self.sceneID)
        # write LST to a geoTiff
        self.ls.clone(lstName, LST)

        print('done processing LST')


def run_rttov(profile_dict):
    nlevels = profile_dict['P'].shape[1]
    nprofiles = profile_dict['P'].shape[0]
    my_profiles = pyrttov.Profiles(nprofiles, nlevels)
    my_profiles.GasUnits = 2
    my_profiles.P = profile_dict['P']
    my_profiles.T = profile_dict['T']
    my_profiles.Q = profile_dict['Q']
    my_profiles.Angles = profile_dict['Angles']
    my_profiles.S2m = profile_dict['S2m']
    my_profiles.Skin = profile_dict['Skin']
    my_profiles.SurfType = profile_dict['SurfType']
    my_profiles.SurfGeom = profile_dict['SurfGeom']
    my_profiles.DateTimes = profile_dict['Datetimes']
    month = profile_dict['Datetimes'][0, 1]

    # ------------------------------------------------------------------------
    # Set up Rttov instance
    # ------------------------------------------------------------------------

    # Create Rttov object for the TIRS instrument
    """" Set the options for each Rttov instance:
    - the path to the coefficient file must always be specified
    - specify paths to the emissivity and BRDF atlas data in order to use
        the atlases (the BRDF atlas is only used for VIS/NIR channels so here
      it is unnecessary for HIRS or MHS)
    - turn RTTOV interpolation on (because input pressure levels differ from
      coefficient file levels)
    - set the verbose_wrapper flag to true so the wrapper provides more
      information
    - enable solar simulations for SEVIRI
    - enable CO2 simulations for HIRS (the CO2 profiles are ignored for
      the SEVIRI and MHS simulations)
    - enable the store_trans wrapper option for MHS to provide access to
      RTTOV transmission structure"""

    sensor_rttov = pyrttov.Rttov()
    #    nchan_tirs = 1
    s = pyrttov.__file__
    env_path = os.sep.join(s.split(os.sep)[:-6])
    # rttov_path = os.path.join(env_path, 'share')
    rttov_coeff_path = os.path.join(os.getcwd(), 'rttov')
    rttov_atlas_path = os.path.join(os.getcwd(), 'rttov')
    rttov_emis_path = os.path.join(rttov_atlas_path, 'emis_data')
    rttov_brdf_path = os.path.join(rttov_atlas_path, 'brdf_data')
    if not os.path.exists(rttov_brdf_path):
        os.makedirs(rttov_emis_path)
        os.makedirs(rttov_brdf_path)
        print("missing atlases")
        print(
            " go to https://www.nwpsaf.eu/site/software/rttov/download/rttov-v11/#Emissivity_BRDF_atlas_data_for_RTTOV_v11")
        print(" to download the HDF5 atlases into emis_data and brdf_data folders in the rttov folder")
    #  KARIM CHANGE THIS COEFFICIENT FOR VIIRS================================
    sensor_rttov.FileCoef = '{}/{}'.format(rttov_coeff_path, "rtcoef_landsat_8_tirs.dat")
    # ==============================================================================
    sensor_rttov.EmisAtlasPath = rttov_emis_path
    sensor_rttov.BrdfAtlasPath = rttov_brdf_path

    sensor_rttov.Options.AddInterp = True
    sensor_rttov.Options.StoreTrans = True
    sensor_rttov.Options.StoreRad2 = True
    sensor_rttov.Options.VerboseWrapper = True

    # Load the instruments:

    try:
        sensor_rttov.loadInst()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error loading instrument(s): {!s}".format(e))
        sys.exit(1)

    # Associate the profiles with each Rttov instance
    sensor_rttov.Profiles = my_profiles
    # ------------------------------------------------------------------------
    # Load the emissivity and BRDF atlases
    # ------------------------------------------------------------------------

    """Load the emissivity and BRDF atlases:
    - load data for August (month=8)
    - note that we only need to load the IR emissivity once and it is
      available for both SEVIRI and HIRS: we could use either the seviriRttov
      or hirsRttov object to do this
    - for the BRDF atlas, since SEVIRI is the only VIS/NIR instrument we can
      use the single-instrument initialisation"""

    sensor_rttov.irEmisAtlasSetup(month)
    # ------------------------------------------------------------------------
    # Call RTTOV
    # ------------------------------------------------------------------------

    """Since we want the emissivity/reflectance to be calculated, the
    SurfEmisRefl attribute of the Rttov objects are left uninitialised:
    That way they will be automatically initialise to -1 by the wrapper

    Call the RTTOV direct model for each instrument:
    no arguments are supplied to runDirect so all loaded channels are
    simulated"""

    try:
        sensor_rttov.runDirect()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))
        sys.exit(1)

    return sensor_rttov


def get_lst(earth_user, earth_pass, meta_fn):
    # ------------------------------------------------------------------------
    # Set up the profile data
    # ------------------------------------------------------------------------
    landsat_temp = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(landsat_temp):
        os.makedirs(landsat_temp)

    meta = landsat_metadata(meta_fn)
    sceneID = meta.LANDSAT_SCENE_ID
    ulLat = meta.CORNER_UL_LAT_PRODUCT
    ulLon = meta.CORNER_UL_LON_PRODUCT
    lrLat = meta.CORNER_LR_LAT_PRODUCT
    lrLon = meta.CORNER_LR_LON_PRODUCT
    landsatDate = meta.DATE_ACQUIRED
    landsatTime = meta.SCENE_CENTER_TIME[:-2]
    d = datetime.strptime('%s%s' % (landsatDate, landsatTime), '%Y-%m-%d%H:%M:%S.%f')
    sza = meta.SUN_ELEVATION
    saa = meta.SUN_AZIMUTH
    tif_file = os.path.join(landsat_temp, '%s_lst.tif' % sceneID)

    landsat = Landsat(meta_fn, username=earth_user,
                      password=earth_pass)

    rttov_obj = rttov(earth_user, earth_pass, ulLat, ulLon, lrLat, lrLon, sza, saa, d)
    if not os.path.exists(tif_file):
        profile_dict = rttov_obj.prepare_profile_data()
        tiirs_rttov = run_rttov(profile_dict)
        landsat.processLandsatLST(tiirs_rttov, profile_dict)


def main():
    # KARIM============
    earth_user = 'user'
    earth_pass = 'pass'
    # KARIM=================
    # =====earthData credentials===============
    # unnecessary for now it simply saves your password
    # if earth_user is None:
    #     earth_user = str(getpass.getpass(prompt="earth login username:"))
    #     if keyring.get_password("nasa", earth_user) is None:
    #         earth_pass = str(getpass.getpass(prompt="earth login password:"))
    #         keyring.set_password("nasa", earth_user, earth_pass)
    #     else:
    #         earth_pass = str(keyring.get_password("nasa", earth_user))

    """ You will need:
    1. Landsat MTL path
    2. put BRDF and Emmisivity atlases in ./rttov/brdf_data/ and ./rttov/emis_data/
    3. put sensor specific coefficients in ./rttov/ (i.e. rtcoef_landsat_8_tirs.dat)
    4. you will need your NASA Earth Data Login"""

    meta_fn = "xxxxx_MTL.txt"
    get_lst(earth_user, earth_pass, meta_fn)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, pycurl.error):
        exit('Received Ctrl + C... Exiting! Bye.', 1)
