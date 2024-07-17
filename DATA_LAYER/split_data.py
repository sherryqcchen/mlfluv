import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random
import numpy as np
import shutil
import argparse
from UTILS import utils

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

def split_n_folds(n, folder_list, save_dir=None, which_label='ESRI'):
    '''
    Split a list of folders into n folds
    Author: QC
    Args:
        n: int, n folds.
        folder_list: list, a list of folder paths where all data points are stored.
        out_fname: the path of npy file that store split information.
    Return:
        
    '''

    print(f'All data folders are split into {n} folds.')
    random.shuffle(folder_list)
    print('The length of all data in the list:', len(folder_list))
 
    paths_per_fold = len(folder_list) // n
    folds = [folder_list[i * paths_per_fold: (i + 1) * paths_per_fold] for i in range(n)]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"{save_dir} created.")

    for i, fold in enumerate(folds):
        # print(fold)
        
        fold_list = []
        for path in fold:
            file_paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('S1.npy') 
                          or file.endswith('S2.npy') 
                          or file.endswith('hand.tif') 
                          or file.endswith(f'{which_label}.npy')]
            print(len(file_paths))
            if len(file_paths)==4:
                print(path)
            file_paths.sort() # Sorted order is: ESRI.npy, ESRI_hand.tif, S1.noy, S2.npy
            fold_list.append(file_paths)
        
        # save the data path split by 5 folds in npy files
        fold_fname = f"fold_{i}.npy"
        fold_path = os.path.join(save_dir, fold_fname)
        np.save(fold_path, fold_list)


def get_s12label_list(WHICH_LABEL, data_path):

    label_list = []
    
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        else:
            continue
        
        s1_path = [file for file in file_paths if file.endswith('S1.npy')][0]
        s2_path = [file for file in file_paths if file.endswith('S2.npy')][0]

        s1_arr = np.load(s1_path)
        s2_arr = np.load(s2_path)
      
        if WHICH_LABEL != 'hand':
            label_path = [file for file in file_paths if file.endswith(f'{WHICH_LABEL}.npy')][0]
            label_arr = np.load(label_path)

            # print('Before removing NaNs:', np.unique(label_arr))

            # Mask -inf in the label as 0
            label_arr[~np.isfinite(label_arr)] = 0

            # Convert invalid data to np.nan
            s1_arr[~np.isfinite(s1_arr)] = np.nan
            s2_arr[(s2_arr<0) | (s2_arr>10000)] = np.nan

            # Masking where NaNs in Sentinel data as 0 in the label
            if np.isnan(s2_arr).any() or np.isnan(s1_arr).any():
                continue
                # mask_s1 = np.isnan(s1_arr)
                # mask_s2 = np.isnan(s2_arr)
                # mask_s1_aggregated = np.any(mask_s1, axis=-1)
                # mask_s2_aggregated = np.any(mask_s2, axis=-1)
                # union_mask = np.logical_or(mask_s1_aggregated, mask_s2_aggregated)

                # label_arr[union_mask] = 0

            np.save(label_path, label_arr)
            # print('After removing NaNs:', np.unique(label_arr))

        label_list.append(folder_path)

    return label_list

def filter_paths_by_folder_names(paths, folder_names):
    """Filter paths to only include those that contain any of the folder names."""
    filtered_paths = []
    for path in paths:
        if not any(folder_name in path for folder_name in folder_names):
            filtered_paths.append(path)
    return filtered_paths

if __name__ == '__main__':

    root_path = ''
    root_path, is_vm = utils.update_root_path_for_machine(root_path=root_path)

    if is_vm:
        config_path = os.path.join(root_path,'script/config_1003.yml')
    else:
        config_path = os.path.join(root_path, 'script/config_k8s.yml')

    ####################################
    # PARSE CONFIG FILE
    ####################################
    parser = argparse.ArgumentParser(description="Please provide a configuration ymal file for trainning a U-Net model.")
    parser.add_argument('--config_path',type=str, default=config_path, help='Path to a configuration yaml file.' )
    parser.add_argument('--split_train_only',type=bool,default=False, help='True if only train data is splited into folds.' )

    args = parser.parse_args()
    config_params = utils.load_config(args.config_path)
    sample_mode = config_params['sample']['sample_mode']
    sample_length = config_params['sample']['sample_length'] 
    WHICH_LABEL = config_params['data_loader']['which_label']
    test_data_path = config_params['data_loader']['test_paths']
    with_extra_urban = config_params["incremental_learning"]['with_extra_urban']

    train_data_path = os.path.join(root_path, 
                                   f'data/clean_data/mlfluv_s12lulc_data_water_from_{sample_mode}_{sample_length}')
    print("Processing train data.")
    # train_label_list = get_s12label_list(WHICH_LABEL, train_data_path)
    # split_n_folds(5, train_label_list, save_dir = os.path.join(root_path, 
    #                                                            f'data/fold_data/{sample_mode}_sampling_{WHICH_LABEL}_5_fold', which_label=WHICH_LABEL))

    if not args.split_train_only:
        # Getting the folder list for sediment and bare class seperation
        print('Processing sediment data.')
        sediment_label_list = get_s12label_list(WHICH_LABEL, os.path.join(root_path, f'data/clean_data/mlfluv_incremental_data_sediment'))
        print('Processing bare data.')
        bare_label_list = get_s12label_list(WHICH_LABEL, os.path.join(root_path, f'data/clean_data/mlfluv_incremental_data_bare'))
        print('Processing urban data.')
        urban_label_list = get_s12label_list(WHICH_LABEL, os.path.join(root_path, f'data/clean_data/mlfluv_incremental_data_urban'))
        # Concatenate lists into one list for incremental learning (fine tuning)
        if with_extra_urban:
            incremental_label_list = sediment_label_list + bare_label_list + urban_label_list
            print('Spliting incremental data: sediment, bareland and urban.')
            split_n_folds(5, incremental_label_list, save_dir=os.path.join(root_path, f'data/fold_data/finetune_with_urban_bare_{WHICH_LABEL}_5_fold'), which_label=WHICH_LABEL)
        else:
            # incremental_label_list = sediment_label_list + bare_label_list
            print('Spliting incremental data: sediment and bareland.')
            # split_n_folds(5, incremental_label_list, save_dir=os.path.join(root_path, f'data/fold_data/finetune_{WHICH_LABEL}_5_fold'), which_label=WHICH_LABEL)
        
        print('Processing test data.')
        test_label_list = get_s12label_list('hand', test_data_path)

        # Get a full list of hand labelled images as test data
        print('Getting one fold of full test data.')
        # split_n_folds(1, test_label_list, save_dir=os.path.join(root_path, f'data/fold_data/test_{WHICH_LABEL}_fold'), which_label=WHICH_LABEL)

        # Split hand labelled images into 5 folds
        print('Getting 5 folds of test data.')
        final_test_folders = os.listdir('/exports/csce/datastore/geos/users/s2135982/MLFLUV_DATA/final_test_data')
        final_test_list = [os.path.join(test_data_path, file) for file in final_test_folders]
        split_n_folds(1, final_test_list, save_dir=os.path.join(root_path, f'data/fold_data/final_test_{WHICH_LABEL}_fold'), which_label=WHICH_LABEL)


        calibrate_label_list = filter_paths_by_folder_names(test_label_list, final_test_folders)
        print(len(calibrate_label_list))
        split_n_folds(4, calibrate_label_list, save_dir=os.path.join(root_path, f'data/fold_data/final_test_{WHICH_LABEL}_4_fold'), which_label=WHICH_LABEL)


