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

def split_n_folds(n, folder_list, save_dir=None, which_label='ESRI'):
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
            file_paths.sort() # Sorted order is: ESRI.npy, ESRI_hand.tif, S1.noy, S2.npy
            # print(file_paths)
            fold_list.append(file_paths)
        
        # save the data path split by 5 folds in npy files
        fold_fname = f"fold_{i}.npy"
        fold_path = os.path.join(save_dir, fold_fname)
        np.save(fold_path, fold_list)
    
    return fold_list

if __name__ == '__main__':

    INITIAL_TRAIN = True
    WHICH_LABEL = 'DW' # DW, ESRI, ESAWC, GLC10

    MODE = 'RANDOM' # RANDOM, STRATIFIED, sediment, bare
    
    data_path = 'projects/MLFLUV/data/clean_data'

    labelled_data_dir = [dir for dir in os.listdir(data_path) if MODE in dir][0]
    labelled_data_path = os.path.join(data_path, labelled_data_dir)

    path_list = [os.path.join(labelled_data_dir, file) for file in os.listdir(labelled_data_path)]

    if not INITIAL_TRAIN:
        # Spliting data for fine-tuning/incremental learning
        MODE = 'sediment_bare'
        sedi_dir = [dir for dir in os.listdir(data_path) if 'sediment' in dir][0]
        bare_dir = [dir for dir in os.listdir(data_path) if 'bare' in dir][0]

        sedi_list = os.listdir(os.path.join(data_path,sedi_dir))
        bare_list = os.listdir(os.path.join(data_path,bare_dir))

        sedi_full_path = [os.path.join(sedi_dir, file) for file in sedi_list]
        bare_full_path = [os.path.join(bare_dir, file) for file in bare_list]

        path_list = sedi_full_path + bare_full_path
    
    print(f'Spliting data for {WHICH_LABEL} labels on {MODE} sampling data.')
    # hand_label_list = glob.glob(os.path.join(labelled_data_path, '**/*hand.tif'))
    # auto_label_list = glob.glob(os.path.join(labelled_data_path, f'**/*{WHICH_LABEL}.npy'))
    # print(len(hand_label_list))

    label_list = []
    # compare_list = []
    for folder in path_list:
        # print(folder)
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        else:
            continue
        
        s1_path = [file for file in file_paths if file.endswith('S1.npy')][0]
        s2_path = [file for file in file_paths if file.endswith('S2.npy')][0]
        label_path = [file for file in file_paths if file.endswith(f'{WHICH_LABEL}.npy')][0]

        s1_arr = np.load(s1_path)
        s2_arr = np.load(s2_path)
        label_arr = np.load(label_path)

        # Convert invalid data to np.nan
        s1_arr[~np.isfinite(s1_arr)] = np.nan
        s2_arr[(s2_arr<0) | (s2_arr>10000)] = np.nan

        # Masking where NaNs in Sentinel data as 0 in the label
        if np.isnan(s2_arr).any() or np.isnan(s1_arr).any():
            mask_s1 = np.isnan(s1_arr)
            mask_s2 = np.isnan(s2_arr)
            mask_s1_aggregated = np.any(mask_s1, axis=-1)
            mask_s2_aggregated = np.any(mask_s2, axis=-1)
            union_mask = np.logical_or(mask_s1_aggregated, mask_s2_aggregated)

            label_arr[union_mask] = 0

            np.save(label_path, label_arr)

        label_list.append(folder_path)
    
    print(len(label_list))

    # Random shuffle data and split them into 5 folds
    save_fold_path = os.path.join('projects/MLFLUV/data/fold_data', f'{MODE}_sampling_{WHICH_LABEL}_5_fold')
    split_n_folds(5, label_list, save_dir=save_fold_path, which_label=WHICH_LABEL)
