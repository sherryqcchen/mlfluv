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
import rioxarray
import xarray as xr

if __name__ == '__main__':

    dir = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/data'
    folders = glob.glob(os.path.join(dir, '*'))

    valid_folders = [folder for folder in folders if len(glob.glob(os.path.join(folder, '*')))==6]
    invalid_folders = [folder for folder in folders if len(glob.glob(os.path.join(folder, '*')))!=6]

    print(f"The size of valid data points that has complete images and metainfo file: {len(valid_folders)}")
    # # Check how many files in the incomplete data folders, if less than 6, it means the data is not completely retrived, and such data point should be abandoned.
    # for folder in invalid_folders:
    #     files = glob.glob(os.path.join(folder, '*'))
    #     print(folder)
    #     print(len(files))

    # check for nan values in the dataset
    clean_list = []
    for folder in valid_folders: 
        has_nan = False
        folder_path = os.path.join(dir, folder)
        fnames = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

        for fname in fnames:
            image = xr.open_dataarray(os.path.join(folder_path, fname))
            if image.isnull().any().data == True:
                print(f"Imae that has Nana: {fname}")
                has_nan = True
                break
        if not has_nan:
            # print(clean_list)
            clean_list.append(folder)
            
    print(f'The size of valid data points that has no nan values: {len(clean_list)}.')


    # Generate a split indices for 5-folds
    s = list(range(1, len(valid_folders)))
    random.shuffle(s)
    s = [s[i::5] for i in range(5)]


    # Record the data split by 5 folds in json file
    n_folds = {}
    with open('fold_split.json', 'w') as f:
        for i, paths in enumerate(s):
            print(f'The fold has {len(paths)} datapoints.')
            fold_key = 'fold_' + str(i)
            n_folds[fold_key] = [valid_folders[i] for i in paths]
        json.dump(n_folds, f)


    # Move data into 5-fold folders
    def move_file(f_list, dest_dir):
        """
        Moving image pairs into given directory
        Args:
            f_list: original directory where images are stored
            dest_dir: destination directory
        """
        for folder in f_list:
            folder_name = os.path.basename(folder)
            print(folder_name)
            for item in os.listdir(folder):
                new_dir = os.path.join(dest_dir, folder_name)
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)
                    print(os.path.join(new_dir, os.path.basename(item)))
                    shutil.move(os.path.join(folder, item), os.path.join(new_dir, os.path.basename(item)))
                else: 
                    shutil.move(os.path.join(folder, item), os.path.join(new_dir, os.path.basename(item)))
    
    with open('fold_split.json', 'r') as file:
        # Load the JSON data into a Python dictionary
        fold_paths = json.load(file)

    keys = list(fold_paths.keys())

    dest_folder = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/fold_data'
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    
    for key in keys:
        print(key)
        dir_list = fold_paths[key]
        sub_dir = os.path.join(dest_folder, key)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
            move_file(dir_list, sub_dir)
        else:
            move_file(dir_list, sub_dir)

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
                continue
            else:
                print(f'The tiffs from folder {fname} needs to unify crs systems.')


