#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:33:18 2017

@author: mschull
"""
import os 
import numpy as np
from datetime import datetime
import subprocess
from osgeo import gdal
import h5py
import shutil
from .landsatTools import landsat_metadata,GeoTIFF
from .utils import folders,writeArray2Tiff,getHTTPdata
from pydap.cas import urs
from pydap import client



class RTTOV:
    def __init__(self, filepath,username,password):
        base = os.getcwd()
        Folders = folders(base)  
        self.earthLoginUser = username
        self.earthLoginPass = password   
        self.landsatSR = Folders['landsat_SR']
        meta = landsat_metadata(filepath)
        self.productID = meta.LANDSAT_PRODUCT_ID
        self.scene = self.productID.split('_')[2]
        self.ulLat = meta.CORNER_UL_LAT_PRODUCT
        self.ulLon = meta.CORNER_UL_LON_PRODUCT
        self.lrLat = meta.CORNER_LR_LAT_PRODUCT
        self.lrLon = meta.CORNER_LR_LON_PRODUCT
        self.solZen = meta.SUN_ELEVATION
        self.solAzi = meta.SUN_AZIMUTH
        d = meta.DATETIME_OBJ
        self.year = d.year
        self.month = d.month
        self.day = d.day
        self.hr = d.hour #UTC

    def preparePROFILEdata(self):

        ul = [self.ulLon-1.5,self.ulLat+1.5]
        lr = [self.lrLon+1.5,self.lrLat-1.5]
        # The data is lat/lon and upside down so [0,0] = [-90.0,-180.0]
        maxX = int((lr[0]-(-180))/0.625)
        minX = int((ul[0]-(-180))/0.625)
        minY = int((lr[1]-(-90))/0.5)
        maxY = int((ul[1]-(-90))/0.5)
        
        
        if self.year <1992:
            fileType = 100
        elif self.year >1991 and self.year < 2001:
            fileType=200
        elif self.year > 2000 and self.year<2011:
            fileType = 300
        else:
            fileType = 400
        
        #Instantaneous Two-Dimensional Collections
        #inst1_2d_asm_Nx (M2I1NXASM): Single-Level Diagnostics
        #=============================================================================
        #'https://goldsmr4.sci.gsfc.nasa.gov/opendap/hyrax/MERRA2/'
        opendap_url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/'
        product = 'M2I1NXASM.5.12.4'
        filename = 'MERRA2_%d.inst1_2d_asm_Nx.%04d%02d%02d.nc4' % (fileType,self.year,self.month,self.day)
        fullUrl =os.path.join(opendap_url,product,'%04d'% self.year,'%02d'% self.month,filename)
        #d=open_dods(fullUrl+'?PS[1:1:23][0:1:360][0:1:575]')
        session = urs.setup_session(username = self.earthLoginUser, 
                            password = self.earthLoginPass,
                            check_url=fullUrl)
        d = client.open_url(fullUrl, session=session)
    #    d.keys()
        #surface presure [Pa]
        surfacePressure=d.PS
        sp = np.squeeze(surfacePressure[self.hr,minY:maxY,minX:maxX]/100) # Pa to kPa
        sprshp =np.reshape(sp,sp.shape[0]*sp.shape[1])
        
        #2m air Temp (K)
        Temp2 = d.T2M
        #Temp2=open_dods(fullUrl+'?T2M[1:1:23][0:1:360][0:1:575]')
        t2 = np.squeeze(Temp2[self.hr,minY:maxY,minX:maxX])
        t2rshp =np.reshape(t2,t2.shape[0]*t2.shape[1])
        
        #2m specific humidity [kg kg -1] -> 2 m water vapor [ppmv]
        spcHum = d.QV2M
        #spcHum=open_dods(fullUrl+'?QV2M[1:1:23][0:1:360][0:1:575]')
        # wv_mmr = 1.e-6 * wv_ppmv_layer * (Rair / Rwater)
        # wv_mmr in kg/kg, Rair = 287.0, Rwater = 461.5
        q = np.squeeze(spcHum[self.hr,minY:maxY,minX:maxX])
        q2 = q/(1e-6*(287.0/461.5))
        q2rshp =np.reshape(q2,q2.shape[0]*q2.shape[1])
        
        # skin temp [K]
        sktIn = d.TS
        #sktIn=open_dods(fullUrl+'?TS[1:1:23][0:1:360][0:1:575]')
        skt = np.squeeze(sktIn[self.hr,minY:maxY,minX:maxX])
        sktrshp =np.reshape(skt,skt.shape[0]*skt.shape[1])
        
        # U10M 10-meter_eastward_wind [m s-1]
        u10In = d.U10M
        #u10In=open_dods(fullUrl+'?U10[1:1:23][0:1:360][0:1:575]')
        u10 = np.squeeze(u10In[self.hr,minY:maxY,minX:maxX])
        u10rshp =np.reshape(u10,u10.shape[0]*u10.shape[1])
        
        # V10M 10-meter_northward_wind [m s-1]
        v10In = d.V10M
        #v10In=open_dods(fullUrl+'?V10M[1:1:23][0:1:360][0:1:575]')
        v10 = np.squeeze(v10In[self.hr,minY:maxY,minX:maxX])
        v10rshp =np.reshape(v10,v10.shape[0]*v10.shape[1])
        
        opendap_url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/'
        product = 'M2I3NVASM.5.12.4'
        filename = 'MERRA2_%d.inst3_3d_asm_Nv.%04d%02d%02d.nc4' % (fileType,self.year,self.month,self.day)
        fullUrl =os.path.join(opendap_url,product,'%04d'% self.year,'%02d'% self.month,filename)
        session = urs.setup_session(username = self.earthLoginUser, 
                    password = self.earthLoginPass,
                    check_url=fullUrl)
        d = client.open_url(fullUrl,session=session)
        hr = int(np.round(self.hr/3.)) # convert from 1 hr to 3 hr dataset
        
        #layers specific humidity [kg kg -1] -> 2 m water vapor [ppmv]
        qvIn = d.QV
        #qvIn=open_dods(fullUrl+'?QV[0:1:7][0,:1:71][0:1:360][0:1:575]')
        # wv_mmr = 1.e-6 * wv_ppmv_layer * (Rair / Rwater)
        # wv_mmr in kg/kg, Rair = 287.0, Rwater = 461.5
        qv = np.squeeze(qvIn[hr,:,minY:maxY,minX:maxX])
        qv = qv/(1e-6*(287.0/461.5))
        qvrshp =np.reshape(qv,[qv.shape[0],qv.shape[1]*qv.shape[2]]).T
        
        
        #layers air temperature [K]
        tIn = d.T
        #tIn=open_dods(fullUrl+'?T[0:1:7][0,:1:71][0:1:360][0:1:575]')
        # wv_mmr = 1.e-6 * wv_ppmv_layer * (Rair / Rwater)
        # wv_mmr in kg/kg, Rair = 287.0, Rwater = 461.5
        t = np.squeeze(tIn[hr,:,minY:maxY,minX:maxX])
        trshp =np.reshape(t,[t.shape[0],t.shape[1]*t.shape[2]]).T
        
        #mid_level_pressure [Pa]
        
        plIn=d.PL
        #plIn=open_dods(fullUrl+'?PL[0:1:7][0,:1:71][0:1:360][0:1:575]')
        pl = np.squeeze(plIn[hr,:,minY:maxY,minX:maxX]/100) # Pa to kPa
        plrshp =np.reshape(pl,[pl.shape[0],pl.shape[1]*pl.shape[2]]).T
        #qrshp =np.reshape(q,q.shape[0]*q.shape[1])
        
        
        LAT = d.lat
        LON = d.lon
        lats = LAT[:]
        lons = LON[:]
        lat = np.tile(lats,(len(lons),1)).T
        latIn = np.squeeze(lat[minY:maxY,minX:maxX])
        latrshp =np.reshape(latIn,latIn.shape[0]*latIn.shape[1])
        lon = np.tile(lons,(len(lats),1))
        lonIn = np.squeeze(lon[minY:maxY,minX:maxX])
        lonrshp =np.reshape(lonIn,lonIn.shape[0]*lonIn.shape[1])
        el = np.repeat(0.0,v10.shape[0]*v10.shape[1]) #NEED DEM
        #check surface pressure
        
        
        sunzen = np.repeat(self.solZen,v10.shape[0]*v10.shape[1])
        sunazi = np.repeat(self.solAzi,v10.shape[0]*v10.shape[1])
        fetch = np.repeat(100000,v10.shape[0]*v10.shape[1])
        satzen = np.repeat(0.0,v10.shape[0]*v10.shape[1])
        satazi = np.repeat(0.0,v10.shape[0]*v10.shape[1])
        
        # Units for gas profiles
        gas_units = 2  # ppmv over moist air
        
        # datetimes[6][nprofiles]: yy, mm, dd, hh, mm, ss
        datetimes = np.tile([self.year, self.month, self.day, hr, 0, 0],(v10.shape[0]*v10.shape[1],1))
        
        # angles[4][nprofiles]: satzen, satazi, sunzen, sunazi
        #get from landsat MTL
        angles = np.vstack((satzen,satazi,sunzen,sunazi)).T
        
        # surftype[2][nprofiles]: surftype, watertype
        surftype = np.zeros([angles.shape[0],2]) #NEED LAND/WATER mask
        
        # surfgeom[3][nprofiles]: lat, lon, elev
        surfgeom = np.vstack((latrshp,lonrshp,el)).T
        
        # s2m[6][nprofiles]: 2m p, 2m t, 2m q, 10m wind u, v, wind fetch
        s2m = np.vstack((sprshp,t2rshp,q2rshp,u10rshp,v10rshp,fetch)).T
        
        # skin[9][nprofiles]: skin T, salinity, snow_frac, foam_frac, fastem_coefsx5
        sal = np.repeat(35.0,v10.shape[0]*v10.shape[1])
        snow_frac = np.repeat(0.0,v10.shape[0]*v10.shape[1])
        foam_frac= np.repeat(0.0,v10.shape[0]*v10.shape[1])
        fastem_coef1 = np.repeat(3.0,v10.shape[0]*v10.shape[1])
        fastem_coef2 = np.repeat(5.0,v10.shape[0]*v10.shape[1])
        fastem_coef3 = np.repeat(15.0,v10.shape[0]*v10.shape[1])
        fastem_coef4 = np.repeat(0.1,v10.shape[0]*v10.shape[1])
        fastem_coef5 = np.repeat(0.3,v10.shape[0]*v10.shape[1])
        
        skin= np.vstack((sktrshp,sal,snow_frac,foam_frac,fastem_coef1,fastem_coef2,fastem_coef3,fastem_coef4,fastem_coef5)).T
    
        outDict = {'P':plrshp,'T':trshp,'Q':qvrshp,'Angles':angles,'S2m':s2m,\
        'Skin': skin,'SurfType':surftype,'SurfGeom':surfgeom,'Datetimes':datetimes,\
        'origShape':t2.shape}
        
        return outDict
    

class Landsat:
    def __init__(self, filepath,username,password):
        base = os.getcwd()
        Folders = folders(base)    
        self.earthLoginUser = username
        self.earthLoginPass = password
        self.landsatLST = Folders['landsat_LST']
        self.landsatSR = Folders['landsat_SR']
        self.landsatTemp = Folders['landsat_Temp']
        self.asterEmissivityBase= Folders['asterEmissivityBase']
        self.ASTERmosaicTemp = Folders['ASTERmosaicTemp']
        self.landsatDataBase = Folders['landsatDataBase']
        self.landsatEmissivityBase = Folders['landsatEmissivityBase']
#        self.sceneID = filepath.split(os.sep)[-1][:-8]
#        self.scene = self.sceneID[3:9]
#        self.yeardoy = self.sceneID[9:16]
        
#        meta = landsat_metadata(os.path.join(self.landsatSR, 
#                                                          self.scene,'%s_MTL.txt' % self.sceneID))
        meta = landsat_metadata(filepath)
        self.productID = meta.LANDSAT_PRODUCT_ID
        self.sceneID = meta.LANDSAT_SCENE_ID
        self.scene = self.productID.split('_')[2]
        self.ls = GeoTIFF(os.path.join(self.landsatSR, self.scene,'%s_sr_band1.tif' % self.productID))
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
        d = datetime.strptime('%s%s' % (self.landsatDate,self.landsatTime),'%Y-%m-%d%H:%M:%S.%f')
        self.year = d.year
        self.month = d.month
        self.day = d.day
        self.hr = d.hour #UTC    
    

    def processASTERemis(self):     
        ASTERurlBase = 'https://e4ftl01.cr.usgs.gov/ASTT/AG100.003/2000.01.01'
        # use Landsat scene area
        latRange = range(int(np.floor(self.lrLat)),int(np.ceil(self.ulLat))+1)
        lonRange = range(int(np.floor(self.ulLon)),int(np.ceil(self.lrLon))+1)
        for i in range(len(latRange)):
            for j in range(len(lonRange)):
                latString = "{1:0{0}d}".format(2 if latRange[i]>=0 else 3,latRange[i])
                lonString = "{1:0{0}d}".format(3 if lonRange[j]>=0 else 4,lonRange[j])
                asterFN = 'AG100.v003.%s.%s.0001.h5' % (latString,lonString)
                # ASTER Emissivity product AG100 comes in 1 x 1 degree tiles where UL is given in the filename.
                ASTERurl = os.path.join(ASTERurlBase,asterFN)
                #print ASTERurl
                localAsterFN = os.path.join(self.asterEmissivityBase,asterFN)
                
                if not os.path.isfile(localAsterFN):
                    print "downloading ASTER..."
                    try:
                        getHTTPdata(ASTERurl,localAsterFN,(self.earthLoginUser,self.earthLoginPass))
                    except Exception:
                        print("failed to get the file")
                        continue          
            #open HDF file, extract the desired dataset and save to GTiff
                
                fh5  = h5py.File(localAsterFN , 'r')
                EmisBand4 = np.array(fh5["/Emissivity/Mean/"])[3]/1000.
                lats = np.array(fh5["/Geolocation/Latitude/"])
                lons = np.array(fh5["/Geolocation/Longitude/"])
                tempName = os.path.join(self.ASTERmosaicTemp,'emis%s%s.tiff'% (latString,lonString))
                writeArray2Tiff(EmisBand4,lats[:,0],lons[0,:],tempName)
       
        #mosaic ,reproject and save as geotiff
        mosaicTempFN = '%s/mosaic.vrt' % self.ASTERmosaicTemp
        mosaicVRTcommand = 'gdalbuildvrt -srcnodata 0 %s %s/*.tiff' % (mosaicTempFN,self.ASTERmosaicTemp)
        out = subprocess.check_output(mosaicVRTcommand, shell=True)
        resampName = os.path.join(self.landsatEmissivityBase,'%s_EMIS.tiff' % self.sceneID)
        command = "gdalwarp -overwrite -s_srs '%s' -t_srs '%s' -r bilinear -tr 30 30 -te %f %f %f %f -of GTiff %s %s" % (self.inProj4,self.proj4,self.ulx,self.lry,self.lrx,self.uly, mosaicTempFN,resampName)
        out = subprocess.check_output(command, shell=True)
        print(out)
        print 'done processing ASTER'
        shutil.rmtree(self.ASTERmosaicTemp)
        os.makedirs(self.ASTERmosaicTemp)
        return resampName
    
 
    
    def processLandsatLST(self,tirsRttov,merraDict):
        
        # Landsat brightness temperature
        landsat = os.path.join(self.landsatTemp,"%s_bt_band10.tif" % self.productID)
        Lg = gdal.Open(landsat)
        BT= Lg.ReadAsArray()/10.
        #=====get radiance from BT=========
        ThermalRad = self.Kappa1/(np.exp(self.Kappa2/BT)-1)
        Lg = None
    
        origShap = merraDict['origShape']
        surfgeom=merraDict['SurfGeom']
        nlevels = merraDict['P'].shape[1]
        

        #reshape and resize image to fit landsat
        lats = np.flipud(np.resize(surfgeom[:,0],origShap))
        lons = np.flipud(np.resize(surfgeom[:,1],origShap))

        channel=1

        nu4 = 1/(10.895*0.0001) # convert to cm
        
        #Process downwelling radiance
        RadDown = np.flipud(np.resize(tirsRttov.Rad2Down[:,channel,nlevels-2],origShap))
        tempName = os.path.join(self.landsatDataBase,'RadDown.tiff')
        resampName = os.path.join('%sReproj.tiff' % tempName[:-4])
        writeArray2Tiff(RadDown,lats[:,0],lons[0,:],tempName)

        command = "gdalwarp -overwrite -s_srs '%s' -t_srs '%s' -r bilinear -tr 30 30 -te %d %d %d %d -of GTiff %s %s" % (self.inProj4,self.proj4,self.ulx,self.lry,self.lrx,self.uly,tempName,resampName)
        out = subprocess.check_output(command, shell=True)
        resampName2 = resampName[:-4]+'2.tiff'
        command = "gdalwarp -overwrite -ts %d %d -of GTiff %s %s" % (ThermalRad.shape[1],ThermalRad.shape[0],resampName,resampName2)        
        out = subprocess.check_output(command, shell=True)
        Lg = gdal.Open(resampName2)
        RadDown = Lg.ReadAsArray()
        RadDown = (RadDown*(nu4**2/10**7))#*.001
        Lg = None
        
        #Process upwelling radiance
        RadUp = np.flipud(np.resize(tirsRttov.Rad2Up[:,channel,nlevels-2],origShap))
        tempName = os.path.join(self.landsatDataBase,'RadUp.tiff')
        resampName = os.path.join('%sReproj.tiff' % tempName[:-4])

        writeArray2Tiff(RadUp,lats[:,0],lons[0,:],tempName)
        command = "gdalwarp -overwrite -s_srs '%s' -t_srs '%s' -r bilinear -tr 30 30 -te %d %d %d %d -of GTiff %s %s" % (self.inProj4,self.proj4,self.ulx,self.lry,self.lrx,self.uly,tempName,resampName)
        out = subprocess.check_output(command, shell=True)
        resampName2 = resampName[:-4]+'2.tiff'
        command = "gdalwarp -overwrite -ts %d %d -of GTiff %s %s" % (ThermalRad.shape[1],ThermalRad.shape[0],resampName,resampName2)        
        out = subprocess.check_output(command, shell=True)
        Lg = gdal.Open(resampName2)
        RadUp = Lg.ReadAsArray()
        RadUp = (RadUp*(nu4**2/10**7))#*.001
        Lg = None
        
        #Process transmission
        trans = np.flipud(np.resize(tirsRttov.TauTotal[:,channel],origShap))
        tempName = os.path.join(self.landsatDataBase,'trans.tiff')
        resampName = os.path.join('%sReproj.tiff' % tempName[:-4])
        writeArray2Tiff(trans,lats[:,0],lons[0,:],tempName)

        command = "gdalwarp -overwrite -s_srs '%s' -t_srs '%s' -r bilinear -tr 30 30 -te %d %d %d %d -of GTiff %s %s" % (self.inProj4,self.proj4,self.ulx,self.lry,self.lrx,self.uly,tempName,resampName)
        out = subprocess.check_output(command, shell=True)
        resampName2 = resampName[:-4]+'2.tiff'
        command = "gdalwarp -overwrite -ts %d %d -of GTiff %s %s" % (ThermalRad.shape[1],ThermalRad.shape[0],resampName,resampName2)        
        out = subprocess.check_output(command, shell=True)
        Lg = gdal.Open(resampName2)
        trans = Lg.ReadAsArray()
        Lg = None
          
        #get emissivity from ASTER
        
        if not os.path.exists(os.path.join(self.landsatEmissivityBase,'%s_EMIS.tiff' % self.sceneID)):
            ASTERemisFNtemp = self.processASTERemis()
            ASTERemisFN = ASTERemisFNtemp[:-4]+'2.tiff'
            command = "gdalwarp -overwrite -ts %d %d -of GTiff %s %s" % (trans.shape[1],trans.shape[0],ASTERemisFNtemp,ASTERemisFN)
            out = subprocess.check_output(command, shell=True)
        else:
            ASTERemisFNtemp = os.path.join(self.landsatEmissivityBase,'%s_EMIS.tiff' % self.sceneID)
            ASTERemisFN = ASTERemisFNtemp[:-4]+'2.tiff'
            command = "gdalwarp -overwrite -ts %d %d -of GTiff %s %s" % (trans.shape[1],trans.shape[0],ASTERemisFNtemp,ASTERemisFN)
            out = subprocess.check_output(command, shell=True)
            
        aster = gdal.Open(ASTERemisFN)
        emis = aster.ReadAsArray()
        aster = None
        # calcualte LST
        emis[emis<0.000001] = np.nan
        surfRad =(((ThermalRad-RadUp)/trans)-(1-emis)*RadDown)/emis
        #get Kappa constants from Landsat

        LST = np.array(self.Kappa2*(1/np.log(self.Kappa1/surfRad)), dtype='float32')

        
        lstName = os.path.join(self.landsatTemp,'%s_lst.tiff'% self.sceneID)
        #write LST to a geoTiff
        self.ls.clone(lstName ,LST)

        
        print('done processing LST')