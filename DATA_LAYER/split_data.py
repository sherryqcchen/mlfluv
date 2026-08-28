import os
import sys
# Add the parent directory to the system path
SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(SCRIPT_ROOT)
import random
import numpy as np
import shutil
import argparse
from UTILS import utils

def find_project_root(start_path):
    start_path = os.path.abspath(start_path)
    current_path = start_path
    while True:
        if os.path.isdir(os.path.join(current_path, "data")):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            return os.getcwd()
        current_path = parent_path

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y")

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

def split_n_folds(n, folder_list, save_dir=None, which_label='ESRI', random_seed=None, overwrite=False):
    '''
    Split a list of folders into n folds
    Author: QC
    Args:
        n: int, n folds.
        folder_list: list, a list of folder paths where all data points are stored.
        save_dir: the path of npy file that store split information.
    Return:
        
    '''

    print(f'All data folders are split into {n} folds.')
    folder_list = list(folder_list)
    if random_seed is not None:
        print(f'Using random seed: {random_seed}')
        rng = random.Random(int(random_seed))
        rng.shuffle(folder_list)
    else:
        random.shuffle(folder_list)
    print('The length of all data in the list:', len(folder_list))
 
    paths_per_fold = len(folder_list) // n
    folds = [folder_list[i * paths_per_fold: (i + 1) * paths_per_fold] for i in range(n)]

    if save_dir:
        if os.path.exists(save_dir) and not overwrite:
            raise FileExistsError(f"{save_dir} already exists. Use a new output directory or set overwrite=True.")
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


def get_s12label_list(WHICH_LABEL, data_path, update_labels=True, s2_source='main'):

    label_list = []
    
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        else:
            continue

        try:
            s1_path = [file for file in file_paths if file.endswith('S1.npy')][0]
            s2_path = [file for file in file_paths if file.endswith('S2.npy')][0]

            s1_arr = np.load(s1_path)
            s2_arr = np.load(s2_path)
        except:
            print(f'Skipping folder due to missing S1 or S2 data: {folder_path}')
            continue

        if s2_source == 'l1c_backup':
            s2_backup_path = os.path.join(os.path.dirname(s2_path), 'temp_backup', os.path.basename(s2_path))
            if not os.path.isfile(s2_backup_path):
                print(f'Skipping folder due to missing L1C backup S2 data: {folder_path}')
                continue
      
        if update_labels and WHICH_LABEL != 'hand':
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
                # continue
                mask_s1 = np.isnan(s1_arr)
                mask_s2 = np.isnan(s2_arr)
                mask_s1_aggregated = np.any(mask_s1, axis=-1)
                mask_s2_aggregated = np.any(mask_s2, axis=-1)
                union_mask = np.logical_or(mask_s1_aggregated, mask_s2_aggregated)

                label_arr[union_mask] = 0

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

    root_path = find_project_root(SCRIPT_ROOT)
    # root_path, is_vm = utils.update_root_path_for_machine(root_path=root_path)

    # if is_vm:
    config_path = os.path.join(SCRIPT_ROOT, 'config.yml')
    # else:
        # config_path = os.path.join(root_path, 'script/config_k8s.yml')

    ####################################
    # PARSE CONFIG FILE
    ####################################
    parser = argparse.ArgumentParser(description="Please provide a configuration ymal file for trainning a U-Net model.")
    parser.add_argument('--config_path',type=str, default=config_path, help='Path to a configuration yaml file.' )
    parser.add_argument('--split_train_only', type=str_to_bool, default=False, help='True if only train data is splited into folds.')

    args = parser.parse_args()
    config_params = utils.load_config(args.config_path)
    root_path = config_params.get('project_root', root_path)
    sample_mode = config_params['sample']['sample_mode']
    sample_length = config_params['sample']['sample_length'] 
    WHICH_LABEL = config_params['data_loader']['which_label']
    test_data_path = config_params['data_loader']['test_paths']
    with_extra_urban = config_params["incremental_learning"]['with_extra_urban']
    data_loader_cfg = config_params['data_loader']
    random_seed = data_loader_cfg.get('random_seed')
    update_labels_on_split = data_loader_cfg.get('update_labels_on_split', False)
    s2_source = data_loader_cfg.get('s2_source', 'main')
    fold_output_suffix = data_loader_cfg.get('fold_output_suffix')
    if fold_output_suffix is None and random_seed is not None:
        fold_output_suffix = f"seed_{random_seed}"

    default_train_data_path = os.path.join(
        root_path,
        f'data/clean_data/mlfluv_s12lulc_data_water_from_{sample_mode}_{sample_length}'
    )
    if not os.path.isdir(default_train_data_path):
        default_train_data_path = os.path.join(root_path, f'data/extra_clean_data/mlfluv_s12lulc_data_clean_{sample_mode}')
    train_data_path = data_loader_cfg.get('initial_train_data_path', default_train_data_path)
    if not os.path.isabs(train_data_path):
        train_data_path = os.path.join(root_path, train_data_path)

    fold_dir_name = f'{sample_mode}_sampling_{WHICH_LABEL}_5_fold'
    if fold_output_suffix:
        fold_dir_name = f'{fold_dir_name}_{fold_output_suffix}'
    train_fold_save_dir = os.path.join(root_path, 'data/fold_data', fold_dir_name)
    print("Processing train data.")
    print(f"Training data path: {train_data_path}")
    print(f"Training fold save dir: {train_fold_save_dir}")
    train_label_list = get_s12label_list(
        WHICH_LABEL,
        train_data_path,
        update_labels=update_labels_on_split,
        s2_source=s2_source
    )
    split_n_folds(5, train_label_list, save_dir=train_fold_save_dir, which_label=WHICH_LABEL, random_seed=random_seed)

    if not args.split_train_only:
        # Getting the folder list for sediment and bare class seperation
        print('Processing sediment data.')
        # sediment_label_list = get_s12label_list(WHICH_LABEL, os.path.join(root_path, f'data/extra_clean_data/mlfluv_s12lulc_data_clean_sediment'))
        print('Processing bare data.')
        # bare_label_list = get_s12label_list(WHICH_LABEL, os.path.join(root_path, f'data/extra_clean_data/mlfluv_s12lulc_data_clean_bare'))
        print('Processing urban data.')
        # urban_label_list = get_s12label_list(WHICH_LABEL, os.path.join(root_path, f'data/extra_clean_data/mlfluv_s12lulc_data_clean_urban'))
        # Concatenate lists into one list for incremental learning (fine tuning)
        if with_extra_urban:
            # incremental_label_list = sediment_label_list + bare_label_list + urban_label_list
            print('Spliting incremental data: sediment, bareland and urban.')
            # split_n_folds(5, incremental_label_list, save_dir=os.path.join(root_path, f'data/fold_data/finetune_with_urban_bare_{WHICH_LABEL}_5_fold'), which_label=WHICH_LABEL)
        else:
            # incremental_label_list = sediment_label_list + bare_label_list
            print('Spliting incremental data: sediment and bareland.')
            # split_n_folds(5, incremental_label_list, save_dir=os.path.join(root_path, f'data/fold_data/finetune_{WHICH_LABEL}_5_fold'), which_label=WHICH_LABEL)
        
        print('Processing test data.')
        test_label_list = get_s12label_list('hand', test_data_path)

        # Get a full list of hand labelled images as test data
        print('Getting one fold of full test data.')
        split_n_folds(1, test_label_list, save_dir=os.path.join(root_path, f'data/fold_data/test_{WHICH_LABEL}_fold'), which_label=WHICH_LABEL)

        # Split hand labelled images into 5 folds
        print('Getting 5 folds of test data.')
        final_test_folders = os.listdir('data/labelled_data/final_test_data')
        # final_test_list = [os.path.join(test_data_path, file) for file in final_test_folders]
        final_label_list = get_s12label_list('hand', 'data/labelled_data/final_test_data')
        split_n_folds(1, final_label_list, save_dir=os.path.join(root_path, f'data/fold_data/final_test_{WHICH_LABEL}_fold'), which_label=WHICH_LABEL)


        calibrate_label_list = filter_paths_by_folder_names(test_label_list, final_test_folders)
        print(len(calibrate_label_list))
        split_n_folds(4, calibrate_label_list, save_dir=os.path.join(root_path, f'data/fold_data/final_test_{WHICH_LABEL}_4_fold'), which_label=WHICH_LABEL)

        compare_test_list = get_s12label_list(WHICH_LABEL, os.path.join(root_path, f'data/extra_clean_data/mlfluv_s12lulc_data_clean_compare'))
        split_n_folds(1, compare_test_list, save_dir=os.path.join(root_path, f'data/fold_data/compare_{WHICH_LABEL}_fold'), which_label=WHICH_LABEL)
