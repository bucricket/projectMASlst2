#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 11:51:03 2017

@author: mschull
"""

import os
import glob
import subprocess
import sys
import pyrttov
import argparse
import pycurl
import keyring
import getpass
import ftplib
from .processData import Landsat,RTTOV
from .utils import folders,untar,getFile
from .lndlst_dms import getSharpenedLST


base = os.getcwd()
Folders = folders(base)   
landsat_SR = Folders['landsat_SR']
landsat_LST = Folders['landsat_LST']
landsat_temp = Folders['landsat_Temp']


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

def get_lst(earth_user,earth_pass):
    sceneIDlist = glob.glob(os.path.join(landsat_temp,'*_MTL.txt'))


    # ------------------------------------------------------------------------
    # Set up the profile data
    # ------------------------------------------------------------------------
    for i in xrange(len(sceneIDlist)):
        inFN = sceneIDlist[i]
        landsat = Landsat(inFN,username = earth_user,
                          password = earth_pass)
        rttov = RTTOV(inFN,username = earth_user,
                          password = earth_pass)
        tifFile = os.path.join(landsat_temp,'%s_lst.tiff'% landsat.sceneID)
        binFile = os.path.join(landsat_temp,"lndsr."+landsat.sceneID+".cband6.bin")
        if not os.path.exists(tifFile):
            profileDict = rttov.preparePROFILEdata()
            tiirsRttov = runRTTOV(profileDict)
            landsat.processLandsatLST(tiirsRttov,profileDict)

        subprocess.call(["gdal_translate","-of", "ENVI", "%s" % tifFile, "%s" % binFile])

    #=====sharpen the corrected LST==========================================

        getSharpenedLST(inFN)
    
    #=====move files to their respective directories and remove temp

        binFN = os.path.join(landsat_temp,'%s.sharpened_band6.bin' % landsat.sceneID)
        tifFN = os.path.join(landsat_LST,'%s_lstSharp.tiff' % landsat.sceneID)
        subprocess.call(["gdal_translate", "-of","GTiff","%s" % binFN,"%s" % tifFN]) 
        
        
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("earth_user", type=str, help="earth Login Username")
    parser.add_argument("earth_pass", type=str, help="earth Login Password")
    args = parser.parse_args()
    earth_user = args.earth_user
    earth_pass = args.earth_pass
    # =====earthData credentials===============
    if earth_user == None:
        earth_user = str(getpass.getpass(prompt="earth login username:"))
        if keyring.get_password("nasa",earth_user)==None:
            earth_pass = str(getpass.getpass(prompt="earth login password:"))
            keyring.set_password("nasa",earth_user,earth_pass)
        else:
            earth_pass = str(keyring.get_password("nasa",earth_user)) 

    get_lst(earth_user,earth_pass)
    


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, pycurl.error):
        exit('Received Ctrl + C... Exiting! Bye.', 1)   