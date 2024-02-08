# This scrip is used to examine and plot all data: S1, S2, ESRI label, ESA label, FROM-GL10 label and DW label.
# The plots are examined to decide which label product is better suited to generate labels for fluvial system. 
# ESRI is used for making the MLFluv labels. All the bareland class pixels are converted to fluvial sediment class. 
# Filter out all the ESRI labels that contains water pixels. We use these labels as the starting point to make hand labels.
# Convert S1, S2 and ESRI label from npy files to tif files, so that they can be opened in QGIS. 
# Fowllowing this script, download Planet images for the same aoi, create hand labels by fixing ESRI labels in QGIS (usnig Thrase plugin).


import shutil
import glob
import os
import json
import geojson
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import plotter

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

def convert_npy_to_tiff(npy_path, which_data, meta_info_path, out_tiff_dir, remap_sediment=False):
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
    
    projection = json.loads(meta_df['projection'][0].replace("\'", "\""))
    crs = projection['crs']

    # This transformation is not correct since it only has a scale, no xshearing, xtranslation, yshearing, ytranslation
    # Check some discussion about it: https://gis.stackexchange.com/questions/443080/how-to-access-image-crs-transform-from-ee-projection-object-in-earth-engine
    transform = projection['transform'] 

    # Get scale (pixel size)
    scale = transform[0]

    aoi_coords = json.loads(meta_df['aoi'][0])
    xmin, xmax, ymin, ymax = get_bounding_box(aoi_coords)
    
    # Fix the transformation
    fixed_transform = from_origin(xmin, ymax, scale, scale)  

    out_path = os.path.join(out_tiff_dir, os.path.basename(npy_path).split('.')[0]+'.tif')

    # Define the output TIFF file path
    if which_data == 's2':
        # Select RGB bands (assuming 0-based indexing)
        rgb_data = arr[:, :, [3, 2, 1]]
        # Write the RGB NumPy array to a GeoTIFF file
        with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=rgb_data.shape[0],
                width=rgb_data.shape[1],
                count=rgb_data.shape[2],  # Number of bands (RGB)
                dtype=rgb_data.dtype,
                crs=crs,
                transform=fixed_transform,
                nodata=-999  # Set if there is a nodata value
        ) as dst:
            dst.write(rgb_data.transpose(2, 0, 1)) 
                
    elif which_data == 's1':
        # Select VV band (assuming 0-based indexing)
        vv_data = s1_arr[:, :, 0]
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
        
    else:
        # For all kinds or labels with dimensions [length, width, 1]
        # Write the label Numpy array to a TIFF file
        if remap_sediment:
            arr[arr == 6] = 5

        label = arr #.squeeze()
        with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=label.shape[0],
                width=label.shape[1],
                count=1,  # Number of bands
                dtype='int32',
                crs=crs,
                transform=fixed_transform,
                nodata=-999  # Set if there is a nodata value
        ) as dst:
            dst.write(label.transpose(2, 0, 1))  # Writing the ndarray to the TIFF file


if __name__=='__main__':

    PLOT_DATA = True
    CONVERT_TO_TIFF = True
    REMAP_SEDIMENT = True

    data_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_s12lulc_data_1000_sample'
    
    point_path_list = glob.glob(os.path.join(data_path, '*'))
    print(f"The count of total downloaded data points: {len(point_path_list)}")
    
    water_point_path = []
    fluvial_point_path = []
    for idx, point_path in enumerate(point_path_list):
        # print(point_path)

        # if '20200106T151701_20200106T151658_T18MWB_325S7405W' in point_path:
        #     print(f'I find this weird point at index {idx}.')
        # else:
        #     continue
        
        point_id = os.path.basename(point_path)
        print(f"{idx}: {point_id}")

        file_paths = [os.path.join(point_path, fname) for fname in os.listdir(point_path)]

        esri_label_path = [file for file in file_paths if file.endswith('ESRI.npy')][0]
        glc10_label_path = [file for file in file_paths if file.endswith('GLC10.npy')][0]
        dw_label_path = [file for file in file_paths if file.endswith('DW.npy')][0]
        esawc_label_path = [file for file in file_paths if file.endswith('ESAWC.npy')][0]

        s1_path = [file for file in file_paths if file.endswith('S1.npy')][0]
        s2_path = [file for file in file_paths if file.endswith('S2.npy')][0]

        meta_path = [file for file in file_paths if file.endswith('.csv')][0]

        esri_arr = np.load(esri_label_path)
        glc10_arr = np.load(glc10_label_path)
        dw_arr = np.load(dw_label_path)
        esawc_arr = np.load(esawc_label_path)

        s1_arr = np.load(s1_path)
        s2_arr = np.load(s2_path)
    
        # Create a mask for invalid data in S2 image, replace invalid data with NaNs
        s2_arr[(s2_arr<0) | (s2_arr>10000)] = np.nan

        # Convert invalid values (NaNs) in S1 image to np.nan
        s1_arr[~np.isfinite(s1_arr)] = np.nan

        # Drop data point that contains nan, continue to next point
        if np.isnan(s2_arr).any() or np.isnan(s1_arr).any():
            continue

        if PLOT_DATA:
            # Plot out all the images to compare how useful 4 global LULC products are. 

            # read meta data of this point
            meta_df = pd.read_csv(meta_path)
            plotter.plot_full_data(s1_arr, s2_arr, esri_arr, esawc_arr, dw_arr, glc10_arr, meta_df, True, point_id)
        
        # Check if ESRI label has any water pixel (its pixel value is 0)
        if np.isin(esri_arr, 0).any():
            water_point_path.append(point_path)


    # copy a list of folders with water pixels to a new directory       
    print(f"The count of data points that have water and bare pixels: {len(water_point_path)}")

    # The path for storing the data with both water and bare pixels
    dest_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_s12lulc_data_water_from_1000_sample'
    
    for path in water_point_path:    

        water_point_id = os.path.basename(path)

        file_paths = [os.path.join(path, fname) for fname in os.listdir(path)]

        esri_label_path = [file for file in file_paths if file.endswith('ESRI.npy')][0]

        s1_path = [file for file in file_paths if file.endswith('S1.npy')][0]
        s2_path = [file for file in file_paths if file.endswith('S2.npy')][0]

        meta_path = [file for file in file_paths if file.endswith('.csv')][0]

        # Create a folder in the new directory
        new_path = os.path.join(dest_path, water_point_id)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        
        # Convert Sentinel-1&2 and ESRI label to tiff and write out in the new fluvial directory
        if CONVERT_TO_TIFF:
            convert_npy_to_tiff(s1_path, 's1', meta_path, new_path)
            convert_npy_to_tiff(s2_path, 's2', meta_path, new_path)
            convert_npy_to_tiff(esri_label_path, 'label', meta_path, new_path, REMAP_SEDIMENT)  # remap_sediment = True      

        for fname in os.listdir(path):
            file_path = os.path.join(path, fname)
            shutil.copy(file_path, new_path)

    fluv_point_path_list = glob.glob(os.path.join(dest_path, '*'))
    
    for path in fluv_point_path_list:

        point_id = os.path.basename(path)

        file_paths = [os.path.join(path, fname) for fname in os.listdir(path)]

        esri_label_path = [file for file in file_paths if file.endswith('ESRI.npy')][0]
        s1_path = [file for file in file_paths if file.endswith('S1.npy')][0]
        s2_path = [file for file in file_paths if file.endswith('S2.npy')][0]

        meta_path = [file for file in file_paths if file.endswith('.csv')][0]

        if REMAP_SEDIMENT:
            # convert bare pixels (6) to sediment label (5) for all images in the new fluvial folder:
            esri_arr = np.load(esri_label_path)
            esri_arr[esri_arr == 6] = 5
            
            new_esri_path = esri_label_path.split('.')[0] + '_water.npy'
            np.save(new_esri_path, esri_arr) 

        # Plot S1, S2, ESRI label
        if PLOT_DATA:
            s1_arr = np.load(s1_path)
            s2_arr = np.load(s2_path)
            meta_df = pd.read_csv(meta_path)

            plotter.plot_s12label(s1_arr, s2_arr, esri_arr, meta_df, True, point_id)

        

        

    # TODO trim data to 512*512 in the dataloader
    

             


