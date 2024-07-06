import os
import pandas as pd
from UTILS import plotter
import numpy as np


data_path = 'data/clean_data/mlfluv_s12lulc_data_water_from_STRATIFIED_6000'
HANDLE_NAN_IN_SENTINEL = True
print(os.listdir(data_path))

point_path_list = [os.path.join(data_path, folder) for folder in os.listdir(data_path)]
print(point_path_list)

for idx, point_path in enumerate(point_path_list):

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

        # handle nans in Sentinel data by masking them as 0 in the LULC maps
        if np.isnan(s2_arr).any() or np.isnan(s1_arr).any():
            if HANDLE_NAN_IN_SENTINEL:
                mask_s1 = np.isnan(s1_arr)
                mask_s2 = np.isnan(s2_arr)
                mask_s1_aggregated = np.any(mask_s1, axis=-1)
                mask_s2_aggregated = np.any(mask_s2, axis=-1)
                union_mask = np.logical_or(mask_s1_aggregated, mask_s2_aggregated)
                
                # Updating masked values as zero
                esri_arr[union_mask] = 0
                dw_arr[union_mask] = 0 
                glc10_arr[union_mask] = 0
                esawc_arr[union_mask] = 0
            else:
                # Drop data has NaNs
                continue

        meta_df = pd.read_csv(meta_path)
        plotter.plot_full_data(s1_arr, s2_arr, esri_arr, esawc_arr, dw_arr, glc10_arr, meta_df, True, point_id)
        