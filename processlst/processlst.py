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
# import pyrttov  # put back in once its fixed
import argparse
import pycurl
import keyring
import getpass
import ftplib
from .processData import Landsat, rttov
from .utils import folders, untar, getFile
from .lndlst_dms import getSharpenedLST
import fnmatch
from .landsatTools import landsat_metadata

base = os.getcwd()
Folders = folders(base)
landsat_SR = Folders['landsat_SR']
landsat_LST = Folders['landsat_LST']
landsat_temp = Folders['landsat_Temp']
landsat_cache = os.path.join(base, "SATELLITE_DATA", "LANDSAT")


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


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def find_already_downloaded(cache_dir):
    matches = []
    for root, dirnames, filenames in os.walk(cache_dir):
        for filename in fnmatch.filter(filenames, '*MTL*'):
            matches.append(os.path.join(root, filename))
    available = [os.path.basename(x) for x in matches]
    available = [x[:-8] for x in available]
    return available


def find_not_processed(downloaded, cache_dir):
    """finds the files that are downloaded but still need to process LST data"""
    # find sat
    sat = downloaded[0].split("_")[0][-1]
    # find scenes
    scenes = [x.split("_")[2] for x in downloaded]
    scenes = list(set(scenes))
    available_list = []
    for scene in scenes:
        path_to_search = os.path.join(cache_dir, 'L%s/%s/LST/*_lstSharp.tif' % (sat, scene))
        available = [os.path.basename(x) for x in
                     glob.glob(path_to_search)]
        available = [x[:-8] for x in available]
        available_list = available_list + available
    for x in available_list:
        if x in downloaded:
            downloaded.remove(x)
    return downloaded


def get_lst(earth_user, earth_pass, atmos_corr=True):
    # sceneIDlist = glob.glob(os.path.join(landsat_temp, '*_MTL.txt'))
    downloaded = find_already_downloaded(landsat_cache)
    productIDs = find_not_processed(downloaded, landsat_cache)
    # ------------------------------------------------------------------------
    # Set up the profile data
    # ------------------------------------------------------------------------
    # for i in xrange(len(sceneIDlist)):
    for productID in productIDs:
        sat_str = productID.split("_")[0][-1]
        scene = productID.split("_")[2]
        folder = os.path.join(landsat_cache, "L%s" % sat_str, scene)
        meta_fn = productID + "_MTL.txt"
        in_fn = os.path.join(folder, "RAW_DATA", meta_fn)
        # in_fn = sceneIDlist[i]
        meta = landsat_metadata(meta_fn)
        sceneID = meta.LANDSAT_SCENE_ID
        tif_file = os.path.join(landsat_temp, '%s_lst.tiff' % sceneID)
        bin_file = os.path.join(landsat_temp, "lndsr." + sceneID + ".cband6.bin")
        if atmos_corr is True:
            landsat = Landsat(in_fn, username=earth_user,
                              password=earth_pass)
            rttov_obj = rttov(in_fn, username=earth_user,
                              password=earth_pass)
            if not os.path.exists(tif_file):
                profile_dict = rttov_obj.prepare_profile_data()
                tiirs_rttov = run_rttov(profile_dict)
                landsat.processLandsatLST(tiirs_rttov, profile_dict)
        else:
            tif_file = os.path.join(folder, "RAW_DATA", productID + "_bt_band10.tif")

        subprocess.call(["gdal_translate", "-of", "ENVI", "%s" % tif_file, "%s" % bin_file])

        # =====sharpen the corrected LST==========================================

        getSharpenedLST(in_fn)

        # =====move files to their respective directories and remove temp
        landsat_LST = os.path.join(folder, 'LST')
        if not os.path.exists(landsat_LST):
            os.makedirs(landsat_LST)
        bin_fn = os.path.join(landsat_temp, '%s.sharpened_band6.bin' % landsat.sceneID)
        tif_fn = os.path.join(landsat_LST, '%s_lstSharp.tif' % landsat.sceneID)
        subprocess.call(["gdal_translate", "-of", "GTiff", "%s" % bin_fn, "%s" % tif_fn])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("earth_user", type=str, help="earth Login Username")
    parser.add_argument("earth_pass", type=str, help="earth Login Password")
    parser.add_argument('-a', '--atmos_corr', nargs='?', type=str, default='y',
                        help=' flag to indicate to use atmospheric correction.')
    args = parser.parse_args()
    earth_user = args.earth_user
    earth_pass = args.earth_pass
    atmos_corr = args.atmos_corr

    if atmos_corr == 'y' or atmos_corr == 'Y':
        atmos_corr = True
    else:
        atmos_corr = False
    # =====earthData credentials===============
    if earth_user is None:
        earth_user = str(getpass.getpass(prompt="earth login username:"))
        if keyring.get_password("nasa", earth_user) is None:
            earth_pass = str(getpass.getpass(prompt="earth login password:"))
            keyring.set_password("nasa", earth_user, earth_pass)
        else:
            earth_pass = str(keyring.get_password("nasa", earth_user))

    get_lst(earth_user, earth_pass, atmos_corr)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, pycurl.error):
        exit('Received Ctrl + C... Exiting! Bye.', 1)
