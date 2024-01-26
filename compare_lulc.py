
import shutil
import glob
import os
import numpy as np
import pandas as pd
import plotter



if __name__=='__main__':

    PLOT_DATA = False

    data_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_s12lulc_data'
    point_path_list = glob.glob(os.path.join(data_path, '*'))
    print(len(point_path_list))
    
    water_point_path = []
    fluvial_point_path = []
    for idx, point_path in enumerate(point_path_list):
        
        point_id = os.path.basename(point_path)
        # print(f"{idx}: {point_id}")

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

        # read meta data of this point
        meta_df = pd.read_csv(meta_path)

        if PLOT_DATA:
            plotter.plot_full_data(s1_arr, s2_arr, esri_arr, esawc_arr, dw_arr, glc10_arr, meta_df, True, point_id)
        
        # Check if ESRI label has any water pixel (its pixel value is 0)
        # if np.isin(esri_arr, 0).any():
        #     water_point_path.append(point_path)

        # Check if ESRI label has any water pixel (its pixel value is 0) and bare pixels (6)
        if np.isin(esri_arr, 0).any() and  np.isin(esri_arr, 6).any():
            fluvial_point_path.append(point_path)


    # copy a list of folders with water pixels to a new directory       
    print(len(fluvial_point_path))
    dest_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_s12lulc_data_fluvial'
    
    for path in fluvial_point_path:    

        water_point_id = os.path.basename(path)
        
        # Create a folder in the new directory
        new_path = os.path.join(dest_path, water_point_id)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        for fname in os.listdir(path):
            file_path = os.path.join(path, fname)
            shutil.copy(file_path, new_path)

    # TODO convert bare pixels to sediment label for all images in the new fluvial folder:
             


