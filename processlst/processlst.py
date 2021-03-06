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
from .processData import Landsat, rttov
from .utils import folders, untar, getFile
from .lndlst_dms import getSharpenedLST

base = os.getcwd()
Folders = folders(base)
landsat_SR = Folders['landsat_SR']
landsat_LST = Folders['landsat_LST']
landsat_temp = Folders['landsat_Temp']


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

    tirs_rttov = pyrttov.Rttov()
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
    env_path = os.sep.join(s.split(os.sep)[:-6])
    rttov_path = os.path.join(env_path, 'share')
    rttov_coeff_path = os.path.join(rttov_path, 'rttov')
    rttov_emis_path = os.path.join(rttov_coeff_path, 'emis_data')
    rttov_brdf_path = os.path.join(rttov_coeff_path, 'brdf_data')
    if not os.path.exists(rttov_brdf_path):
        print("downloading atlases.....")
        ftp = ftplib.FTP("ftp.star.nesdis.noaa.gov")
        ftp.login("anonymous", "")

        ftp.cwd('/pub/smcd/emb/mschull/')  # change directory to /pub/
        getFile(ftp, 'rttov_atlas.tar')

        ftp.quit()
        untar('rttov_atlas.tar', rttov_path)

        subprocess.check_output("chmod 755 %s%s*.H5" % (rttov_emis_path, os.sep), shell=True)
        subprocess.check_output("chmod 755 %s%s*.H5" % (rttov_brdf_path, os.sep), shell=True)
    tirs_rttov.FileCoef = '{}/{}'.format(rttov_coeff_path, "rtcoef_landsat_8_tirs.dat")
    tirs_rttov.EmisAtlasPath = rttov_emis_path
    tirs_rttov.BrdfAtlasPath = rttov_brdf_path

    tirs_rttov.Options.AddInterp = True
    tirs_rttov.Options.StoreTrans = True
    tirs_rttov.Options.StoreRad2 = True
    tirs_rttov.Options.VerboseWrapper = True

    # Load the instruments:

    try:
        tirs_rttov.loadInst()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error loading instrument(s): {!s}".format(e))
        sys.exit(1)

    # Associate the profiles with each Rttov instance
    tirs_rttov.Profiles = my_profiles
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

    tirs_rttov.irEmisAtlasSetup(month)
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
        tirs_rttov.runDirect()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))
        sys.exit(1)

    return tirs_rttov


def get_lst(earth_user, earth_pass):
    sceneIDlist = glob.glob(os.path.join(landsat_temp, '*_MTL.txt'))

    # ------------------------------------------------------------------------
    # Set up the profile data
    # ------------------------------------------------------------------------
    for i in xrange(len(sceneIDlist)):
        in_fn = sceneIDlist[i]
        landsat = Landsat(in_fn, username=earth_user,
                          password=earth_pass)
        rttov = rttov(in_fn, username=earth_user,
                      password=earth_pass)
        tif_file = os.path.join(landsat_temp, '%s_lst.tiff' % landsat.sceneID)
        bin_file = os.path.join(landsat_temp, "lndsr." + landsat.sceneID + ".cband6.bin")
        if not os.path.exists(tif_file):
            profile_dict = rttov.prepare_profile_data()
            tiirs_rttov = run_rttov(profile_dict)
            landsat.processLandsatLST(tiirs_rttov, profile_dict)

        subprocess.call(["gdal_translate", "-of", "ENVI", "%s" % tif_file, "%s" % bin_file])

        # =====sharpen the corrected LST==========================================

        getSharpenedLST(in_fn)

        # =====move files to their respective directories and remove temp

        bin_fn = os.path.join(landsat_temp, '%s.sharpened_band6.bin' % landsat.sceneID)
        tif_fn = os.path.join(landsat_LST, '%s_lstSharp.tiff' % landsat.sceneID)
        subprocess.call(["gdal_translate", "-of", "GTiff", "%s" % bin_fn, "%s" % tif_fn])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("earth_user", type=str, help="earth Login Username")
    parser.add_argument("earth_pass", type=str, help="earth Login Password")
    args = parser.parse_args()
    earth_user = args.earth_user
    earth_pass = args.earth_pass
    # =====earthData credentials===============
    if earth_user is None:
        earth_user = str(getpass.getpass(prompt="earth login username:"))
        if keyring.get_password("nasa", earth_user) is None:
            earth_pass = str(getpass.getpass(prompt="earth login password:"))
            keyring.set_password("nasa", earth_user, earth_pass)
        else:
            earth_pass = str(keyring.get_password("nasa", earth_user))

    get_lst(earth_user, earth_pass)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, pycurl.error):
        exit('Received Ctrl + C... Exiting! Bye.', 1)
