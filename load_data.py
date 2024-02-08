import json
import os
import glob
import rasterio
import xarray as xr
import numpy as np
import rioxarray
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

if __name__ == '__main__':
    
    labelled_data_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_s12lulc_data_water_from_1000_sample_labelled'

    # hand_label_list = glob.glob(os.path.join(labelled_data_path, '**/*hand.tif'))
    # print(len(hand_label_list))

    # broken_data_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/labelled_data_with_NaNs'
    label_list = []
    # compare_list = []
    for folder in os.listdir(labelled_data_path):
        print(folder)
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

            continue
        else:
            label_list.append(folder_path)
    
    print(len(label_list))




                

                
                
            

    # train_paths = data['fold_0'] + data['fold_1'] + data['fold_2'] # size 111

    # val_paths = data['fold_3'] # size 36

    # test_paths = data['fold_4'] # size 36

    # print(len(train_paths)) 

    # print(train_paths[0])

    # # Get label from a train data point folder
    # files = os.listdir(train_paths[0])
    # print(files)
    # label_fname = [file for file in files if file.startswith('DWlabel')][0]
    # clear_s2_fname = [file for file in files if file.startswith('clearS2')][0]
    # label_path = os.path.join(train_paths[0], label_fname)
    # clear_s2_path = os.path.join(train_paths[0], clear_s2_fname)
    
    # label = xr.open_dataarray(label_path)
    # clear_s2 = xr.open_dataarray(clear_s2_path)

    # print(label.crs)

    



