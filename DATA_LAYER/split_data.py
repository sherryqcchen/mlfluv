import os
import glob
import random
import numpy as np
import shutil

def move_files(src_dir, dest_dir):
    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Get the absolute path of the source file
            src_file = os.path.join(root, file)
            
            # Get the relative path of the source file with respect to src_dir
            rel_path = os.path.relpath(src_file, src_dir)
            
            # Construct the destination file path
            dest_file = os.path.join(dest_dir, rel_path)
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            
            # Move the file to the destination directory
            shutil.move(src_file, dest_file)

def delete_folder(folder_path):
    try:
        # Attempt to delete the folder and all its contents
        shutil.rmtree(folder_path)
        print(f"Successfully deleted {folder_path}")
    except Exception as e:
        print(f"Error deleting {folder_path}: {e}")

def split_n_folds(n, folder_list, save_dir=None):
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
    paths_per_fold = len(folder_list) // n
    folds = [folder_list[i * paths_per_fold: (i + 1) * paths_per_fold] for i in range(n)]

    # s = list(range(1, len(folder_list)))
    random.shuffle(folds)
    # s = [s[i::n] for i in range(n)]

    # save the data path split by 5 folds in npy files
    if save_dir is None:
        save_dir = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_5_folds'
    for i, fold in enumerate(folds):
        # print(fold)
        
        fold_list = []
        for path in fold:
            file_paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('S1.npy') 
                          or file.endswith('S2.npy') 
                          or file.endswith('hand.tif') 
                          or file.endswith('ESRI.npy')]
            file_paths.sort() # Sorted order is: ESRI.npy, ESRI_hand.tif, S1.noy, S2.npy
            fold_list.append(file_paths)

        fold_fname = f"fold_{i}.npy"
        fold_path = os.path.join(save_dir, fold_fname)
        np.save(fold_path, fold_list)


if __name__ == '__main__':
    
    # labelled_data_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_s12lulc_data_water_from_1000_sample_labelled'
    labelled_data_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_s12lulc_data_water_from_10000_sample'

    # hand_label_list = glob.glob(os.path.join(labelled_data_path, '**/*hand.tif'))
    auto_label_list = glob.glob(os.path.join(labelled_data_path, '**/*ESRI.tif'))
    print(len(auto_label_list))

    # broken_data_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/labelled_data_with_NaNs'
    label_list = []
    # compare_list = []
    for folder in os.listdir(labelled_data_path):
        # print(folder)
        folder_path = os.path.join(labelled_data_path, folder)
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        
        s1_path = [file for file in file_paths if file.endswith('S1.npy')][0]
        s2_path = [file for file in file_paths if file.endswith('S2.npy')][0]
        
        has_nan = False
        s1_arr = np.load(s1_path)
        s2_arr = np.load(s2_path)

        # Convert invalid data to np.nan
        s1_arr[~np.isfinite(s1_arr)] = np.nan
        s2_arr[(s2_arr<0) | (s2_arr>10000)] = np.nan


        # label_path = [file for file in file_paths if file.endswith('hand.tif')][0]
        # label_arr = rasterio.open(label_path).read()
        # if np.isnan(label_arr).any() == True:
        #     has_nan = True
        #     print("Label has NaN.")

        
        if np.isnan(s1_arr).any() == True:
            has_nan = True
            print("S1 has NaN. Delete this data.")

        if np.isnan(s2_arr).any() == True:
            has_nan = True
            print("S2 has NaN. Delete this data.")

        if has_nan == True:
            # # Option 1: Move labelled data with NaNs to a new place 
            # new_path = os.path.join(broken_data_path, folder)
            # os.makedirs(new_path)
            # move_files(folder_path, new_path)

            # # Option 2: Delete this data folder and all the content 
            # delete_folder(folder_path)
            
            # Option 3: ignore this data point, continue to examine next data point
            continue
        else:
            label_list.append(folder_path)
    
    print(len(label_list))

    # Random shuffle data and split them into 5 folds
    # split_n_folds(5, label_list, save_dir='/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_5_folds')

    split_n_folds(5, label_list, save_dir='/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_5_folds_auto')
