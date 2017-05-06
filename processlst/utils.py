# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import urllib2, base64
import os
from osgeo import gdal,osr
import tarfile
import numpy as np



class earthDataHTTPRedirectHandler(urllib2.HTTPRedirectHandler):
    def http_error_302(self, req, fp, code, msg, headers):
        return urllib2.HTTPRedirectHandler.http_error_302(self, req, fp, code, msg, headers)
    

def getHTTPdata(url,outFN,auth=None):
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


def writeArray2Tiff(data,lats,lons,outfile):
    Projection = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    xres = lons[1] - lons[0]
    yres = lats[1] - lats[0]

    ysize = len(lats)
    xsize = len(lons)

    ulx = lons[0] #- (xres / 2.)
    uly = lats[0]# - (yres / 2.)
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(outfile, xsize, ysize, 1, gdal.GDT_Float32)
    
    srs = osr.SpatialReference()
    if isinstance(Projection, basestring):        
        srs.ImportFromProj4(Projection)
    else:
        srs.ImportFromEPSG(Projection)        
    ds.SetProjection(srs.ExportToWkt())
    
    gt = [ulx, xres, 0, uly, 0, yres ]
    ds.SetGeoTransform(gt)
    
    outband = ds.GetRasterBand(1)
    outband.WriteArray(data)    
    ds.FlushCache()  
    
    ds = None
    
def writeArray2Envi(data,ulx,uly,xres,yres,Projection,outfile):
    #Projection = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    if len(data.shape)>2:
        nbands = data.shape[0]
        nrows = data.shape[1]
        ncols = data.shape[2]
    else:
        nbands = 1
        nrows = data.shape[0]
        ncols = data.shape[1]
    
    driver = gdal.GetDriverByName('ENVI')
    ds = driver.Create(outfile, ncols, nrows, nbands, gdal.GDT_Float32)
    
    srs = osr.SpatialReference()
    if isinstance(Projection, basestring):        
        srs.ImportFromProj4(Projection)
    else:
        srs.ImportFromEPSG(Projection)        
    ds.SetProjection(srs.ExportToWkt())
    
    gt = [ulx, xres, 0, uly, 0, yres ]
    ds.SetGeoTransform(gt)
    
    if nbands>1:
        for band in range(nbands):
            ds.GetRasterBand(band+1).WriteArray( data[band,:,:] )
    else:
        ds.GetRasterBand(1).WriteArray( data )
    ds = None


def folders(base):
    dataBase = os.path.join(base,'data')
    landsatDataBase = os.path.join(dataBase,'Landsat-8')
    asterDataBase = os.path.join(dataBase,'ASTER')
    landsat_SR = os.path.join(landsatDataBase,'SR')
    if not os.path.exists(landsat_SR):
        os.makedirs(landsat_SR)
    landsat_Temp = os.path.join(landsat_SR,'temp')
    if not os.path.exists(landsat_Temp):
        os.makedirs(landsat_Temp)
    landsat_LST = os.path.join(landsatDataBase,'LST')
    if not os.path.exists(landsat_LST):
        os.makedirs(landsat_LST)
    asterEmissivityBase = os.path.join(asterDataBase,'asterEmissivity')
    if not os.path.exists(asterEmissivityBase):
        os.makedirs(asterEmissivityBase)
    landsatEmissivityBase = os.path.join(asterDataBase,'landsatEmissivity')
    if not os.path.exists(landsatEmissivityBase):
        os.makedirs(landsatEmissivityBase)
    ASTERmosaicTemp = os.path.join(asterDataBase,'mosaicTemp')    
    if not os.path.exists(ASTERmosaicTemp):
        os.makedirs(ASTERmosaicTemp)
    out = {'landsat_LST':landsat_LST,'landsat_SR':landsat_SR,
    'asterEmissivityBase':asterEmissivityBase,'ASTERmosaicTemp':ASTERmosaicTemp,
    'landsatDataBase':landsatDataBase, 'landsatEmissivityBase':landsatEmissivityBase,
    'landsat_Temp':landsat_Temp}
    return out

def clean(directory,ext):
    test=os.listdir(directory)

    for item in test:
        if item.startswith(ext):
            os.remove(os.path.join(directory, item))

 
def getFile(ftp, filename):
    try:
        ftp.retrbinary("RETR " + filename ,open(filename, 'wb').write)
    except:
        print "Error"
 
            
def untar(fname, fpath):

    tar = tarfile.open(fname)
    tar.extractall(path = fpath)
    tar.close()
    os.remove(fname)
    
    
# helper function
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

