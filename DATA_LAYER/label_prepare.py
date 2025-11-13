# This scrip is used to examine and plot all data: S1, S2, ESRI label, ESA label, FROM-GL10 label and DW label.
# The plots are examined to decide which label product is better suited to generate labels for fluvial system. 
# ESRI is used for making the MLFluv labels. All the bareland class pixels are converted to fluvial sediment class. 
# Filter out all the ESRI labels that contains water pixels. We use these labels as the starting point to make hand labels.
# Convert S1, S2 and ESRI label from npy files to tif files, so that they can be opened in QGIS. 
# Fowllowing this script, download Planet images for the same aoi, create hand labels by fixing ESRI labels in QGIS (usnig Thrase plugin).
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import shutil
import glob
import json
import geojson
import argparse
import numpy as np
import pandas as pd
from UTILS import plotter
from UTILS import utils

import rasterio
from rasterio.transform import from_origin


def get_bounding_box(coord_list):
    """
    Method from https://gis.stackexchange.com/questions/313011/convert-geojson-object-to-bounding-box-using-python-3-6
    Extract bounding box from a list of coordinates.

    Args:
        coord_list (list): geometry coordinates exported from GEE

    Returns:
        list: minimum and maximum longitudes and latitudes of the geometry
    """
    coords = np.array(list(geojson.utils.coords(coord_list)))
    xmin = coords[:, 0].min() # minimum longitude
    xmax = coords[:, 0].max() # maximum longitude
    ymin = coords[:, 1].min() # minimum latitude
    ymax = coords[:, 1].max() # maximum latitude
    return xmin, xmax, ymin, ymax

def convert_npy_to_tiff(npy_path, which_data, meta_info_path, out_tiff_dir, remap_to_sedi=False):
    """
    Convert numpy ndarray from .npy file to tiff for visualization and labelling.

    Args:
        npy_path (str): path to npy file.
        which_data (str): 's1', 's2' or 'label'.
        meta_info_path (str): path to meta information csv file.
        out_tiff_dir (str): the derectory of exporting tiff.
    """ 
    arr = np.load(npy_path)
    meta_df = pd.read_csv(meta_info_path)
    
    # Safely load projection and CRS
    projection = json.loads(meta_df['projection'][0].replace("\'", "\""))
    crs = projection.get('crs')
    transform = projection.get('transform') 

    # Get scale (pixel size)
    scale = transform[0] if transform else 10 # Default scale if transform is missing

    aoi_coords = json.loads(meta_df['aoi'][0])
    xmin, xmax, ymin, ymax = get_bounding_box(aoi_coords)
    
    # Fix the transformation using the extent and scale
    fixed_transform = from_origin(xmin, ymax, scale, scale)  

    out_path = os.path.join(out_tiff_dir, os.path.basename(npy_path).split('.')[0]+'.tif')

    # Define the output TIFF file path
    if which_data == 's2':
        # Write all Sentinel-2 bands (assumes arr shape is (H, W, B))
        s2_data = arr
        with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=s2_data.shape[0],
                width=s2_data.shape[1],
                count=s2_data.shape[2],  # All bands
                dtype=s2_data.dtype,
                crs=crs,
                transform=fixed_transform,
                nodata=-999  # Set if there is a nodata value
        ) as dst:
            dst.write(s2_data.transpose(2, 0, 1))
                
    elif which_data == 's1':
        # Select VV band (assuming 0-based indexing)
        vv_data = arr[:, :, 0]
        # Write the VV NumPy array to a GeoTIFF file
        with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=vv_data.shape[0],
                width=vv_data.shape[1],
                count=1,  # Number of bands (VV)
                dtype=vv_data.dtype,
                crs=crs,
                transform=fixed_transform,
                nodata=-999  # Set if there is a nodata value
        ) as dst:
            dst.write(vv_data, 1)  # Writing the VV ndarray to the TIFF file
        
    else: # Labels
        # For all kinds or labels with dimensions [length, width, 1]
        
        if remap_to_sedi:
            # remap bare class (5) to sediment class (6)
            label = np.where(arr==5, 6, arr)
        else:
            label = arr

        # Ensure label array is 2D for writing a single-band TIFF (squeeze it)
        label = label.squeeze() 

        with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=label.shape[0],
                width=label.shape[1],
                count=1,  # Number of bands
                dtype='int32', # Labels are typically integer types
                crs=crs,
                transform=fixed_transform,
                nodata=-999  # Set if there is a nodata value
        ) as dst:
            # dst.write() expects (1, H, W) for single band, so we need to expand dims
            dst.write(label[np.newaxis, :, :])


if __name__=='__main__':

    root_path = ''
    root_path, is_vm = utils.update_root_path_for_machine(root_path=root_path)

    if is_vm:
        config_path = os.path.join(root_path,'script/config.yml')
    else:
        config_path = os.path.join(root_path, 'script/config_k8s.yml')

    parser = argparse.ArgumentParser(description="Please provide a configuration ymal file for preprocessing labels: functions include plotting, converting .npy to .tif, remapping bare pixels to sediment, handling NaNs in Sentinel and moving cleaned data to a new folder.")
    parser.add_argument('--config_path',type=str, default=config_path, help='Path to a configuration yaml file.' )
    
    args = parser.parse_args()
    config = utils.load_config(args.config_path)

    PLOT_DATA = config['plot_setting']['plot_data']
    CONVERT_TO_TIFF = config['data_preprocess']['convert_to_tif']
    REMAP_TO_SEDI =  config['data_preprocess']['remap_to_sedi']
    HANDLE_NAN_IN_SENTINEL = config['data_preprocess']['handle_nan_in_sentinel']
    FLUV_POINT_ONLY = config['data_preprocess']['fluv_point_only']
    print(FLUV_POINT_ONLY)
    MOVE_DATA = config['data_preprocess']['move_data']

    WHICH_LABEL = config['data_loader']['which_label']
    SAMPLE_MODE = 'compare' # config['sample']['sample_mode'] 

    raw_data_path = os.path.join(root_path,f'data/full_data/')
    
    # Safely find the data folder based on SAMPLE_MODE
    data_folder_list = [file for file in os.listdir(raw_data_path) if SAMPLE_MODE in file]
    if not data_folder_list:
        print(f"Error: No data folder found in {raw_data_path} matching SAMPLE_MODE '{SAMPLE_MODE}'.")
        sys.exit(1)
        
    data_folder = data_folder_list[0]
    data_path = os.path.join(raw_data_path, data_folder)
    
    point_path_list = glob.glob(os.path.join(data_path, '*'))
    print(f"The count of total downloaded data points: {len(point_path_list)}")
    
    fluvial_point_path = []
    
    for idx, point_path in enumerate(point_path_list):

        point_id = os.path.basename(point_path)
        print(f"{idx}: {point_id}")
        
        # Skip if it's the temp_backup folder itself, just in case
        if point_id == 'temp_backup':
            print(f"Skipping temporary directory: {point_path}")
            continue

        file_paths = [os.path.join(point_path, fname) for fname in os.listdir(point_path)]

        # --- Load paths for processing ---
        try:
            dw_label_path = [file for file in file_paths if file.endswith('DW.npy')][0]
            s1_path = [file for file in file_paths if file.endswith('S1.npy')][0]
            s2_path = [file for file in file_paths if file.endswith('S2.npy')][0]
            meta_path = [file for file in file_paths if file.endswith('.csv')][0]
        except IndexError:
            print(f"Skipping point {point_id}: Missing one or more required files (DW.npy, S1.npy, S2.npy, or .csv).")
            continue
            
        # --- Load data ---
        dw_arr = np.load(dw_label_path)
        s1_arr = np.load(s1_path)
        s2_arr = np.load(s2_path)

        if REMAP_TO_SEDI:
            # remap bare class (5) to sediment class (6)
            dw_arr = np.where(dw_arr==5, 6, dw_arr)

        # Create a mask for invalid data in S2 image, replace invalid data with NaNs
        s2_arr[(s2_arr<0) | (s2_arr>10000)] = np.nan

        # Convert invalid values (NaNs) in S1 image to np.nan
        s1_arr[~np.isfinite(s1_arr)] = np.nan

        # handle nans in Sentinel data by masking them as 0 in the LULC maps
        if np.isnan(s2_arr).any() or np.isnan(s1_arr).any():
            if HANDLE_NAN_IN_SENTINEL:
                mask_s1 = np.isnan(s1_arr)
                mask_s2 = np.isnan(s2_arr)
                mask_s1_aggregated = np.any(mask_s1, axis=-1)
                mask_s2_aggregated = np.any(mask_s2, axis=-1)
                union_mask = np.logical_or(mask_s1_aggregated, mask_s2_aggregated)
                
                # Updating masked values as zero in the DW label
                dw_arr[union_mask] = 0 
            else:
                # Drop data has NaNs
                continue

        # Save the potentially updated DW array
        np.save(dw_label_path, dw_arr)

        # --- Filter Fluvial Points ---
        if FLUV_POINT_ONLY:
            # Check if Dynamic World label has any bare pixel (pixel value is 5, already remapped to 6 if REMAP_TO_SEDI is True)
            # The check should be for the target sediment class (6) after remapping, or 5 if no remapping.
            target_class = 6 if REMAP_TO_SEDI else 5
            
            if np.isin(dw_arr, target_class).any():
                fluvial_point_path.append(point_path)
            else:
                continue
        else:
            fluvial_point_path.append(point_path)

        # Define the file name for the list of points to be moved
        point_list_filename = os.path.join(root_path, f'script/DATA_LAYER/{SAMPLE_MODE}_general_points.txt')


    print(f"\nThe count of data points that are selected (fluvial/bare): {len(fluvial_point_path)}")

    # Export a list of fluvial points path to txt file
    fluvial_point_paths = [path + '\n' for path in fluvial_point_path]

    with open(point_list_filename, 'w') as f:
        f.writelines(fluvial_point_paths)

    # --- Setup Destination Path ---
    dest_path = os.path.join(os.path.join(root_path,'data/extra_clean_data'), f'mlfluv_s12lulc_data_clean_{SAMPLE_MODE}')
    
    # Ensure the main destination directory exists
    os.makedirs(dest_path, exist_ok=True)


    # --- FINAL MOVE/COPY & TIFF CONVERSION ---
    
    # Read the confirmed list of paths to be moved
    with open(point_list_filename, 'r') as f:
        paths = f.readlines()

    move_paths = [path.strip() for path in paths]

    for source_point_path in move_paths:  

        water_point_id = os.path.basename(source_point_path)
        dest_point_path = os.path.join(dest_path, water_point_id)

        # 1. Create the destination folder
        if not os.path.exists(dest_point_path):
            os.makedirs(dest_point_path)
        
        # 2. Copy the contents of the point folder (excluding 'temp_backup')
        if MOVE_DATA:
            try:
                # This copies the directory contents from source_point_path to dest_point_path,
                # ignoring any sub-directory or file named 'temp_backup'.
                shutil.copytree(
                    src=source_point_path,
                    dst=dest_point_path,
                    ignore=shutil.ignore_patterns('temp_backup'),
                    dirs_exist_ok=True # Needed since dest_point_path is created above
                )
            except Exception as e:
                print(f"Could not copy directory {source_point_path}: {e}")
                continue # Skip to the next point if copy fails

        # 3. Convert Sentinel and Label to TIFF in the new directory
        # We must use the file paths from the *new* destination to ensure we get the latest processed files.
        if CONVERT_TO_TIFF:
            # Find the file paths in the newly copied directory
            dest_file_paths = [os.path.join(dest_point_path, fname) for fname in os.listdir(dest_point_path)]
            
            dw_label_path = [file for file in dest_file_paths if file.endswith('DW.npy')][0]
            s1_fluv_path = [file for file in dest_file_paths if file.endswith('S1.npy')][0]
            s2_fluv_path = [file for file in dest_file_paths if file.endswith('S2.npy')][0]
            meta_path = [file for file in dest_file_paths if file.endswith('.csv')][0]

            # Convert to TIFF
            convert_npy_to_tiff(s1_fluv_path, 's1', meta_path, dest_point_path)
            convert_npy_to_tiff(s2_fluv_path, 's2', meta_path, dest_point_path)
            # remap_to_sedi=False here because the remapping was already done and saved to the .npy file above.
            convert_npy_to_tiff(dw_label_path, 'label', meta_path, dest_point_path, remap_to_sedi=False)


    # --- PLOTTING ---
    if PLOT_DATA:

        fluv_point_path_list = glob.glob(os.path.join(dest_path, '*'))
        
        for path in fluv_point_path_list:

            point_id = os.path.basename(path)

            file_paths = [os.path.join(path, fname) for fname in os.listdir(path)]
            
            # Use DW.npy as the source label, regardless of WHICH_LABEL config
            label_path = [file for file in file_paths if file.endswith('DW.npy')][0]
            s1_path = [file for file in file_paths if file.endswith('S1.npy')][0]
            s2_path = [file for file in file_paths if file.endswith('S2.npy')][0]

            meta_path = [file for file in file_paths if file.endswith('.csv')][0]

            # Plot S1, S2, DW label
            s1_arr = np.load(s1_path)
            s2_arr = np.load(s2_path)
            label_arr = np.load(label_path)
            meta_df = pd.read_csv(meta_path)

            plotter.plot_s12label(s1_arr, s2_arr, label_arr, meta_df, True, point_id, 'DW')
