import argparse
import os
import shutil
import sys

import yaml

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import glob
import requests
import io
import ee
import numpy as np
import pandas as pd
import ast
from get_ee_data import convert_ee_image_to_np_arr, interpolator

config_path = 'script/config.yml'
with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

try:
    service_account = config['ee_service_account']
    credentials = ee.ServiceAccountCredentials(service_account, config['ee_api_key'])
    ee.Initialize(credentials)
except:
    # If you don't have Earth Engine Service Account Credentials, using following lines
    ee.Initialize()

data_folder_path = "data/labelled_data_S2L1C/final_test_data"

data_path_list = [os.path.join(data_folder_path, folder) for folder in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, folder))]
print(len(data_path_list), "folders found in", data_folder_path)

def download_sentinel2_image(data_folder_path):
    """Replace original Sentinel-2 L1C images with Sentinel-2 L2A images from Google Earth Engine.
        using the metadata CSV files in each folder to get the image IDs and area of interest (AOI).
    """

    print(f"Processing folder: {data_folder_path}")
    
    # 1. Find the old S2 image file
    s2_pattern = os.path.join(data_folder_path, '*S2.npy')
    try:
        # Get the full path to the old S2 file
        old_s2_file_path = next(glob.iglob(s2_pattern))
        old_s2_file_name = os.path.basename(old_s2_file_path)
    except StopIteration:
        print(f"Error: No *S2.npy file found in {data_folder_path}. Skipping.")
        return

    # 2. Define and create the temporary folder
    temp_folder = os.path.join(data_folder_path, 'temp_backup')
    os.makedirs(temp_folder, exist_ok=True)
    print(f"Temporary backup folder created/exists at: {temp_folder}")

    # 3. Move the old S2 image to the temporary folder
    # We use shutil.move for robustness, handles cross-device moves if necessary
    backup_file_path = os.path.join(temp_folder, old_s2_file_name)
    shutil.move(old_s2_file_path, backup_file_path)
    print(f"Moved old S2 image to: {backup_file_path}")
    
    # The new file path will be the original location of the old file
    new_s2_file_path = old_s2_file_path # This is the target path for the new image

    # --- Earth Engine Logic to get new S2 image ---
    
    # Get S2 ID from the folder name (as in your original code)
    # Assuming folder name is something like T10TEK_20200801_0
    folder_name = os.path.basename(data_folder_path)
    s2_id = '_'.join(folder_name.split('_')[:3])
    print(f"Extracted S2 ID: {s2_id}")

    # Find and read the metadata file
    meta_pattern = os.path.join(data_folder_path, '*meta.csv')
    try:
        meta_file = next(glob.iglob(meta_pattern))
    except StopIteration:
        print(f"Error: No *meta.csv file found in {data_folder_path}. Skipping.")
        # Optionally move the backup back if the process stops here, but for now, just return.
        return

    meta_dict = pd.read_csv(meta_file).to_dict('records')[0]
    
    # Handle potential missing 'aoi' or format errors
    try:
        coordinate_list_of_rings = ast.literal_eval(meta_dict['aoi'])
        # Assuming the AOI is a single polygon defined by the first element
        aoi = ee.Geometry.Polygon(coordinate_list_of_rings[0])
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error processing AOI from meta file: {e}. Skipping.")
        return

    # Earth Engine Image Collection filtering (S2 L2A)
    s2_bands = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterMetadata('system:index', 'equals', s2_id) \
        .select('B.*') \
        .first()

    # Get bands from config
    S2_BANDS = config['sample']['s2_bands']
    
    # Convert Earth Engine image to numpy array
    s2_data, s2_proj = convert_ee_image_to_np_arr(s2_bands, S2_BANDS, aoi)
    
    # Apply interpolation
    s2_filled_data = interpolator(s2_data)

    # 4. Save the new S2 image to the original file path
    # The original path is stored in new_s2_file_path (which was old_s2_file_path)
    np.save(new_s2_file_path, s2_filled_data)
    print(f"Successfully downloaded and saved new S2 image to: {new_s2_file_path}")

def check_folder_completeness(folder_path: str) -> bool:
    """
    Checks if a folder path contains the required files for a complete download.
    
    A download is complete if:
    1. A new *S2.npy file exists in the main folder (the downloaded image).
    2. A *S2.npy file exists in the 'temp_backup' subdirectory (the old L1C image).
    """
    # 1. Check for the new S2 output file (*S2.npy directly in the folder)
    final_file_pattern = os.path.join(folder_path, '*S2.npy')
    is_final_file_present = len(glob.glob(final_file_pattern)) > 0

    # 2. Check for the backup file (temp_backup/*S2.npy in a subdirectory)
    backup_file_pattern = os.path.join(folder_path, 'temp_backup', '*S2.npy')
    is_backup_file_present = len(glob.glob(backup_file_pattern)) > 0
    
    # Return True only if both conditions are met
    return is_final_file_present and is_backup_file_present

def find_resume_index(paths: list, start_after_index: int) -> int:
    """
    Iterates through the list starting from the specified index and finds the 
    first path where the download condition is not met.
    """
    start_check_from = start_after_index + 1
    
    if start_check_from >= len(paths):
        print("Starting index is beyond the end of the list. All folders might be processed.")
        return len(paths) # Stop processing
        
    print("-" * 60)
    print(f"RESUMPTION CHECK: Starting from index {start_check_from}...")
    print("-" * 60)
    
    for index in range(start_check_from, len(paths)):
        folder_path = paths[index]
        
        if not check_folder_completeness(folder_path):
            print(f"\n>>> INTERRUPTED DOWNLOAD FOUND at index: {index}")
            print(f"    Path: {folder_path}")
            return index # Return the index where the download failed/stopped
            
    print("\n" + "=" * 60)
    print("RESUMPTION CHECK COMPLETE: All folders checked were complete.")
    print("You can stop the script or start from the beginning if needed.")
    print("=" * 60)
    return len(paths) # Return list length if all are complete

def restore_L1C(data_folder_path):
    """
    Reverses the Sentinel-2 swap by moving the L2A image to a 'level2A_backup' folder 
    and restoring the L1C image from 'temp_backup' to the main folder.
    """
    print(f"\nREVERSING SWAP and organizing backups for folder: {data_folder_path}")
    
    # Define paths
    L1C_source_folder = os.path.join(data_folder_path, 'temp_backup')
    L2A_target_folder = os.path.join(data_folder_path, 'level2A_backup')
    
    # --- 1. Identify Files ---
    
    # The L2A image is currently in the main folder
    l2a_pattern = os.path.join(data_folder_path, '*S2.npy')
    l2a_file_list = glob.glob(l2a_pattern)
    
    # The L1C image is currently in the L1C_source_folder ('temp_backup')
    l1c_pattern = os.path.join(L1C_source_folder, '*S2.npy')
    l1c_file_list = glob.glob(l1c_pattern)
    
    if not l2a_file_list or not l1c_file_list:
        print(f"---Skipping: Cannot find L2A in main folder or L1C in backup for {data_folder_path}.")
        return

    l2a_file_path = l2a_file_list[0] # Path of the current L2A file
    l1c_file_path = l1c_file_list[0] # Path of the current L1C backup file
    s2_file_name = os.path.basename(l2a_file_path) # e.g., T10TEK_S2.npy
    
    # --- 2. Move L2A to New Backup Folder ---
    os.makedirs(L2A_target_folder, exist_ok=True)
    l2a_backup_path = os.path.join(L2A_target_folder, s2_file_name)
    
    try:
        # Move the current L2A image from the main folder to 'level2A_backup'
        shutil.move(l2a_file_path, l2a_backup_path)
        print(f"Moved L2A image to: {l2a_backup_path}")
        
    except Exception as e:
        print(f"---Error moving L2A to backup for {data_folder_path}: {e}")
        return # Stop if the L2A move fails

    # --- 3. Restore L1C from Old Backup Folder ---
    
    # Move the original L1C image from 'temp_backup' back to the main folder
    l1c_restore_path = os.path.join(data_folder_path, s2_file_name)
    try:
        shutil.move(l1c_file_path, l1c_restore_path)
        print(f"Restored L1C image to main folder: {l1c_restore_path}")
        
        # Optional: Remove the now-empty 'temp_backup' folder if it's empty
        try:
            os.rmdir(L1C_source_folder)
            print(f"Removed empty folder: {L1C_source_folder}")
        except OSError:
            # Folder might not be empty if other files were moved there
            pass
            
        print(f"---Swap reversal and organization **COMPLETE** for {data_folder_path}.")
        
    except Exception as e:
        print(f"---Error restoring L1C to main folder for {data_folder_path}: {e}")

for folder_path in data_path_list:
    print(folder_path)
    restore_L1C(folder_path)
    # download_sentinel2_image(folder_path)

# item_to_find = "data/clean_data/mlfluv_s12lulc_data_water_from_STRATIFIED_6000/20200907T151709_20200907T152155_T18MWE_61S7437W"

# # find interrupt index if downloading is interrupted
# try:
#     index = data_path_list.index(item_to_find)
#     print(f"The item's index is: {index}")
# except ValueError:
#     print(f"The item was not found in the list.")



# The 'folder_path' loop you mentioned can then be used to access items:
# for folder_path in data_path_list:
#     ...

# LAST_SUCCESSFUL_INDEX = 2396 

# Find the index where the download should resume (the first incomplete folder)
# resume_index = find_resume_index(data_path_list, LAST_SUCCESSFUL_INDEX)
# print(f"Resuming downloads from index: {resume_index}")