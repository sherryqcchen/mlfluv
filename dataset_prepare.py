# Author: Qiuyang Chen   09/01/2024
# This scrip is for filtering data points after data is downloaded from Earth Engine. 
# It filters out the incomplete data points first, and then clean the data points with NaNs
# All the valid data points are splited into 5 folds for machine learning.
# The splited data paths are stored in a json file.


import os
import json
import glob
import random
import shutil
import numpy as np
import rioxarray
import xarray as xr


def clean_raw_data(dir):
    '''
    1. Remove folders that does not have 6 files: 2 Sentinel-1 tifs, 2 Sentinel-2 tifs, 1 DW label and 1 meta info csv.
    2. Check Nan values in the Sentinel images and labels. Remove folders that contain images with Nan value (usually from Sentinel-1).
    Authir: QC
    Args: 
        dir: directory path to the raw data, where all folders are stored under this dir. Each folder contains (theoratically) 6 files. 
    Return:
        clean_list: a list of valid folder names after cleaning
    '''
    folders = glob.glob(os.path.join(dir, '*'))

    valid_folders = [folder for folder in folders if len(glob.glob(os.path.join(folder, '*')))==6]
    invalid_folders = [folder for folder in folders if len(glob.glob(os.path.join(folder, '*')))!=6]

    # Check how many files in the incomplete data folders, if less than 6, it means the data is not completely retrived, and such data point should be abandoned.
    # for folder in invalid_folders:
    #     files = glob.glob(os.path.join(folder, '*'))
    #     print(f"{folder} contains {len(files)} files: not valid.")

    print(f"The size of valid data points that has complete images and metainfo file: {len(valid_folders)}")

    # check for nan values in the dataset
    clean_list = []
    for folder in valid_folders: 
        has_nan = False
        folder_path = os.path.join(dir, folder)
        fnames = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

        for fname in fnames:
            image = xr.open_dataarray(os.path.join(folder_path, fname))
            if image.isnull().any().data == True:
                # print(f"Imae that has Nan: {fname}")
                has_nan = True
                break
        if not has_nan:
            clean_list.append(folder)
            
    print(f'The size of valid data points that has no Nan values: {len(clean_list)}.')

    return clean_list

def move_file(f_list, dest_dir):
    """
    Move files in a list of folders into a given directory, while retain its old directory structure
    Author: QC
    Args:
        f_list: a list of folders where each folder stores Sentinel-1&2 images, label and metadata csv files for a sample point
        dest_dir: destination directory
    """
    for folder in f_list:
        folder_name = os.path.basename(folder)
        for item in os.listdir(folder):
            new_dir = os.path.join(dest_dir, folder_name)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
                shutil.move(os.path.join(folder, item), os.path.join(new_dir, os.path.basename(item)))
            else: 
                shutil.move(os.path.join(folder, item), os.path.join(new_dir, os.path.basename(item)))


def edit_DW_label(DW_label_path):
    '''
    Modify DW labels: merge classes of shallow-root vegetation (grass, crop, shrub) to one class shallow-root vegetation (with value 2)
    Author: QC
    Args:
        DW_label_path: the full path of a DW label
    Return:
        new_label: the modified label, stored under the same directory of an original DW label.
    '''
    label = xr.open_dataarray(DW_label_path)

    # Merge classes of shallow-root vegetation (grass, crop, shrub) to one class shallow-root vegetation (with value 2) in the DW label: 
    merged_label = xr.where((label==4) | (label==5), 2, label)

    # Re-arrange label values to water=0, tree=1, shallow-root vegetation=2, flooded vegetation=3, build=4, bare=5, snow/ice = 6
    new_label = merged_label.where(label!=6, 4).where(label!=7, 5).where(label!=8, 6)

    save_dir = os.path.dirname(DW_label_path)
    new_label_fname = os.path.basename(DW_label_path)[2:]

    new_label.rio.set_crs(label.rio.crs)
    new_label.rio.to_raster(os.path.join(save_dir, new_label_fname))

    return new_label


def split_n_folds(n, folder_list, out_fname='fold_split.json'):
    '''
    Split a list of folders into n folds
    Author: QC
    Args:
        n: int, n folds.
        folder_list: list, a list of folder paths where all data points are stored.
        out_fname: the path of json file that store split information.
    Return:
        n_folds: dictionary, the key is named as 'fold_n', and the value under each key is a list of folder paths that are stored under this fold.
    '''

    print(f'All data folders are split into {n} folds.')
    s = list(range(1, len(folder_list)))
    random.shuffle(s)
    s = [s[i::n] for i in range(n)]

    # Record the data split by 5 folds in json file
    n_folds = {}
    with open(out_fname, 'w') as f:
        for i, paths in enumerate(s):
            fold_key = 'fold_' + str(i)
            print(f'{fold_key} has {len(paths)} datapoints.')
            n_folds[fold_key] = [folder_list[i] for i in paths]
        json.dump(n_folds, f)

    return n_folds


if __name__ == '__main__':

    dir = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/data'
    raw_data = glob.glob(os.path.join(dir, '*'))
    print(f"The size of raw data is {len(dir)}.")

    # Clean incomplete data and data with Nan value
    clean_folders = clean_raw_data(dir)

    # Split data to 5 folds
    # Generate a split indices for 5-folds
    fold_paths = split_n_folds(5, clean_folders)

    # with open('fold_split.json', 'r') as file:
    #     # Load the JSON data into a Python dictionary
    #     fold_paths = json.load(file)

    # Move the data into splited folds 
    dest_folder = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/fold_data'

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    
    keys = list(fold_paths.keys())

    for key in keys:
        dir_list = fold_paths[key]
        sub_dir = os.path.join(dest_folder, key)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
            move_file(dir_list, sub_dir)
        else:
            move_file(dir_list, sub_dir)

    # Modify DW label classes and store the modification into a new label
    # Check if five tiff files for each data point are using the same crs system
    for folder in os.listdir(dest_folder):
        # print(folder)
        path = os.path.join(dest_folder, folder)
        for fname in os.listdir(path):
            sub_path = os.path.join(path, fname)
            all_tiff = [file for file in os.listdir(sub_path) if file.endswith('.tif')]
            # print(all_tiff)
            all_crs = [rioxarray.open_rasterio(os.path.join(sub_path, file)).rio.crs for file in all_tiff]
            if len(set(all_crs)) == 1:                
                label_fname = [file for file in os.listdir(sub_path) if file.startswith('DWlabel')][0]
                new_label = edit_DW_label(os.path.join(sub_path, label_fname))
            else:
                print(f'The tiffs from folder {fname} needs to unify crs systems.')




