

import glob
import os
import numpy as np
import pandas as pd
import plotter



if __name__=='__main__':

    data_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_s12lulc_data'
    point_path_list = glob.glob(os.path.join(data_path, '*'))
    # print(len(point_path_list))
    for point_path in point_path_list[:20]:

        point_id = os.path.basename(point_path)

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
        # print()

        plotter.plot_full_data(s1_arr, s2_arr, esri_arr, esawc_arr, dw_arr, glc10_arr, meta_df, True, point_id)
