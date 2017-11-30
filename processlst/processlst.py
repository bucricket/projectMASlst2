#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 11:51:03 2017

@author: mschull
"""

import os
import subprocess
import sys
import pyrttov
import argparse
import pycurl
import keyring
import getpass
import types
import copy_reg
import ftplib
import pandas as pd
import sqlite3
from .processData import Landsat,RTTOV
from .utils import folders,untar,getFile
from .lndlst_dms import getSharpenedLST
from getlandsatdata import getlandsatdata
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.DEBUG)



base = os.getcwd()
cacheDir = os.path.abspath(os.path.join(base,os.pardir,"SATELLITE_DATA"))
Folders = folders(base)   
#landsat_LST = Folders['landsat_LST']
landsat_temp = Folders['landsat_Temp']

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

def updateLandsatProductsDB(landsatDB,filenames,cacheDir,product):
    
    db_fn = os.path.join(cacheDir,"landsat_products.db")
    
    date = landsatDB.acquisitionDate
    ullat = landsatDB.upperLeftCornerLatitude
    ullon = landsatDB.upperLeftCornerLongitude
    lllat = landsatDB.lowerRightCornerLatitude
    lllon = landsatDB.lowerRightCornerLongitude
    productIDs = landsatDB.LANDSAT_PRODUCT_ID
    
    if not os.path.exists(db_fn):
        conn = sqlite3.connect( db_fn )
        landsat_dict = {"acquisitionDate":date,"upperLeftCornerLatitude":ullat,
                      "upperLeftCornerLongitude":ullon,
                      "lowerRightCornerLatitude":lllat,
                      "lowerRightCornerLongitude":lllon,
                      "LANDSAT_PRODUCT_ID":productIDs,"filename":filenames}
        landsat_df = pd.DataFrame.from_dict(landsat_dict)
        landsat_df.to_sql("%s" % product, conn, if_exists="replace", index=False)
        conn.close()
    else:
        conn = sqlite3.connect( db_fn )
        res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = res.fetchall()[0]
        if (product in tables):
            orig_df = pd.read_sql_query("SELECT * from %s" % product,conn)
        else:
            orig_df = pd.DataFrame()
            
        landsat_dict = {"acquisitionDate":date,"upperLeftCornerLatitude":ullat,
                      "upperLeftCornerLongitude":ullon,
                      "lowerRightCornerLatitude":lllat,
                      "lowerRightCornerLongitude":lllon,
                      "LANDSAT_PRODUCT_ID":productIDs,"filename":filenames}
        landsat_df = pd.DataFrame.from_dict(landsat_dict)
        orig_df = orig_df.append(landsat_df,ignore_index=True)
        orig_df = orig_df.drop_duplicates(keep='last')
        orig_df.to_sql("%s" % product, conn, if_exists="replace", index=False)
        conn.close()

def searchLandsatProductsDB(lat,lon,start_date,end_date,product,cacheDir):
    db_fn = os.path.join(cacheDir,"landsat_products.db")
    conn = sqlite3.connect( db_fn )

    out_df = pd.read_sql_query("SELECT * from %s WHERE (acquisitionDate >= '%s')"
                   "AND (acquisitionDate < '%s') AND (upperLeftCornerLatitude > %f )"
                   "AND (upperLeftCornerLongitude < %f ) AND "
                   "(lowerRightCornerLatitude < %f) AND "
                   "(lowerRightCornerLongitude > %f)" % 
                   (product,start_date,end_date,lat,lon,lat,lon),conn)   
    conn.close()
    return out_df

def runRTTOV(profileDict):
    nlevels = profileDict['P'].shape[1]
    nprofiles = profileDict['P'].shape[0]
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)
    myProfiles.GasUnits = 2
    myProfiles.P = profileDict['P']
    myProfiles.T = profileDict['T']
    myProfiles.Q = profileDict['Q']
    myProfiles.Angles = profileDict['Angles']
    myProfiles.S2m = profileDict['S2m']
    myProfiles.Skin = profileDict['Skin']
    myProfiles.SurfType = profileDict['SurfType']
    myProfiles.SurfGeom =profileDict['SurfGeom']
    myProfiles.DateTimes = profileDict['Datetimes']
    month = profileDict['Datetimes'][0,1]

    # ------------------------------------------------------------------------
    # Set up Rttov instance
    # ------------------------------------------------------------------------

    # Create Rttov object for the TIRS instrument

    tirsRttov = pyrttov.Rttov()
#    nchan_tirs = 1

    # Set the options for each Rttov instance:
    # - the path to the coefficient file must always be specified
    # - specify paths to the emissivity and BRDF atlas data in order to use
    #   the atlases (the BRDF atlas is only used for VIS/NIR channels so here
    #   it is unnecessary for HIRS or MHS)
    # - turn RTTOV interpolation on (because input pressure levels differ from
    #   coefficient file levels)
    # - set the verbose_wrapper flag to true so the wrapper provides more
    #   information
    # - enable solar simulations for SEVIRI
    # - enable CO2 simulations for HIRS (the CO2 profiles are ignored for
    #   the SEVIRI and MHS simulations)
    # - enable the store_trans wrapper option for MHS to provide access to
    #   RTTOV transmission structure
    s = pyrttov.__file__
    envPath = os.sep.join(s.split(os.sep)[:-6])
    rttovPath = os.path.join(envPath,'share')
    rttovCoeffPath = os.path.join(rttovPath,'rttov')
    rttovEmisPath = os.path.join(rttovCoeffPath,'emis_data')
    rttovBRDFPath = os.path.join(rttovCoeffPath,'brdf_data')
    if not os.path.exists(rttovBRDFPath):
        print("downloading atlases.....")
        ftp = ftplib.FTP("ftp.star.nesdis.noaa.gov")
        ftp.login("anonymous", "")
         
        ftp.cwd('/pub/smcd/emb/mschull/')         # change directory to /pub/
        getFile(ftp,'rttov_atlas.tar')
         
        ftp.quit()
        untar('rttov_atlas.tar',rttovPath)

        subprocess.check_output("chmod 755 %s%s*.H5" % (rttovEmisPath,os.sep), shell=True)   
        subprocess.check_output("chmod 755 %s%s*.H5" % (rttovBRDFPath,os.sep), shell=True)  
    tirsRttov.FileCoef = '{}/{}'.format(rttovCoeffPath,"rtcoef_landsat_8_tirs.dat")
    tirsRttov.EmisAtlasPath = rttovEmisPath 
    tirsRttov.BrdfAtlasPath = rttovBRDFPath 


    tirsRttov.Options.AddInterp = True
    tirsRttov.Options.StoreTrans = True
    tirsRttov.Options.StoreRad2 = True
    tirsRttov.Options.VerboseWrapper = True


    # Load the instruments:

    try:
        tirsRttov.loadInst()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error loading instrument(s): {!s}".format(e))
        sys.exit(1)

    # Associate the profiles with each Rttov instance
    tirsRttov.Profiles = myProfiles
    # ------------------------------------------------------------------------
    # Load the emissivity and BRDF atlases
    # ------------------------------------------------------------------------

    # Load the emissivity and BRDF atlases:
    # - load data for August (month=8)
    # - note that we only need to load the IR emissivity once and it is
    #   available for both SEVIRI and HIRS: we could use either the seviriRttov
    #   or hirsRttov object to do this
    # - for the BRDF atlas, since SEVIRI is the only VIS/NIR instrument we can
    #   use the single-instrument initialisation

    tirsRttov.irEmisAtlasSetup(month)
    # ------------------------------------------------------------------------
    # Call RTTOV
    # ------------------------------------------------------------------------

    # Since we want the emissivity/reflectance to be calculated, the
    # SurfEmisRefl attribute of the Rttov objects are left uninitialised:
    # That way they will be automatically initialise to -1 by the wrapper

    # Call the RTTOV direct model for each instrument:
    # no arguments are supplied to runDirect so all loaded channels are
    # simulated
    try:
        tirsRttov.runDirect()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))
        sys.exit(1)
        
    return tirsRttov

def get_lst(loc,start_date,end_date,earth_user,earth_pass,cloud,sat,cacheDir):
    landsatCacheDir = os.path.join(cacheDir,"LANDSAT")
    db_fn = os.path.join(landsatCacheDir,"landsat_products.db")
    available = 'Y'
    product = 'LST'

    search_df = getlandsatdata.search(loc[0],loc[1],start_date,end_date,cloud,available,landsatCacheDir,sat)
    productIDs = search_df.LANDSAT_PRODUCT_ID
    print(productIDs)
    paths = search_df.local_file_path 
    #====check what products are processed against what Landsat data is available===
    if os.path.exists(db_fn):
        conn = sqlite3.connect( db_fn )
        res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = res.fetchall()[0]
        if (product in tables):
            processedProductIDs = searchLandsatProductsDB(loc[0],loc[1],start_date,end_date,product,landsatCacheDir)
            df1 = processedProductIDs[["LANDSAT_PRODUCT_ID"]]
            merged = df1.merge(pd.DataFrame(productIDs), indicator=True, how='outer')
            df3 = merged[merged['_merge'] != 'both' ]
            productIDs = df3[["LANDSAT_PRODUCT_ID"]].LANDSAT_PRODUCT_ID
            if len(productIDs)>0:
                output_df = pd.DataFrame()
                for productID in productIDs:
                    output_df = output_df.append(getlandsatdata.searchProduct(productID,cacheDir,sat),ignore_index=True)
                paths = output_df.local_file_path
                productIDs = search_df.LANDSAT_PRODUCT_ID
            
                

    # ------------------------------------------------------------------------
    # Set up the profile data
    # ------------------------------------------------------------------------
    print(productIDs)
    fns = []
    for i in range(len(productIDs)):
#        output_df = pd.DataFrame()
#        for productID in productIDs:
#            output_df = output_df.append(getlandsatdata.searchProduct(productID,landsatCacheDir,sat),ignore_index=True)
        productIDpath = os.path.join(paths[i],productIDs[i])
        landsat = Landsat(productIDpath,username = earth_user,
                          password = earth_pass)
        rttov = RTTOV(productIDpath,username = earth_user,
                          password = earth_pass)
        tifFile = os.path.join(landsat_temp,'%s_lst.tiff'% landsat.sceneID)
        binFile = os.path.join(landsat_temp,"lndsr."+landsat.sceneID+".cband6.bin")
        if not os.path.exists(tifFile):
#            profileDict = rttov.preparePROFILEdataCFSR()
            profileDict = rttov.preparePROFILEdata()
            tiirsRttov = runRTTOV(profileDict)
            landsat.processLandsatLST(tiirsRttov,profileDict)

            subprocess.call(["gdal_translate","-of", "ENVI", "%s" % tifFile, "%s" % binFile])
    
        #=====sharpen the corrected LST========================================
    
#            fns.append(getSharpenedLST(productIDpath,sat))
            
        
        #=====move files to their respective directories and remove temp
#    
#            binFN = os.path.join(landsat_temp,'%s.sharpened_band6.bin' % landsat.sceneID)
#            tifFN = os.path.join(landsat_LST,'%s_lstSharp.tiff' % landsat.sceneID)
#            subprocess.call(["gdal_translate", "-of","GTiff","%s" % binFN,"%s" % tifFN]) 
    updateLandsatProductsDB(search_df,fns,landsatCacheDir,'LST')
        
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("lat", type=float, help="latitude")
    parser.add_argument("lon", type=float, help="longitude")
    parser.add_argument("start_date", type=str, help="Start date yyyy-mm-dd")
    parser.add_argument("end_date", type=str, help="Start date yyyy-mm-dd")
    parser.add_argument("cloud", type=int, help="cloud coverage")
    parser.add_argument('-s','--sat', nargs='?',type=int, default=8, help='which landsat to search or download, i.e. Landsat 8 = 8')
    args = parser.parse_args()
      
    loc = [args.lat,args.lon] 
    start_date = args.start_date
    end_date = args.end_date
    cloud = args.cloud
    sat = args.sat

    # =====earthData credentials===============

    earth_user = str(getpass.getpass(prompt="earth login username:"))
    if keyring.get_password("nasa",earth_user)==None:
        earth_pass = str(getpass.getpass(prompt="earth login password:"))
        keyring.set_password("nasa",earth_user,earth_pass)
    else:
        earth_pass = str(keyring.get_password("nasa",earth_user)) 

    get_lst(loc,start_date,end_date,earth_user,earth_pass,cloud,sat,cacheDir)
    


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, pycurl.error):
        exit('Received Ctrl + C... Exiting! Bye.', 1)   