import glob
import os
import numpy as np
import pandas as pd
import rioxarray
import plotter

REMAP_HAND_LABEL = False

data_path = '/exports/csce/datastore/geos/users/s2135982/MLFLUV_DATA/unlabelled_data'

point_path_list = glob.glob(os.path.join(data_path, '*'))
print(f"The count of total downloaded data points: {len(point_path_list)}")

def remap_mask(mask_arr):
    #[water, tree, grass, crop, urban, sedi, bare, nans] to [nans, water, tree, grass, crop, urban, sedi, bare]
    mask_arr = np.where(mask_arr < 0, -1, mask_arr)
    mask_arr = np.where(mask_arr == 7, -1, mask_arr)
    mask_arr = np.where(mask_arr == 5, 7, mask_arr)
    mask_arr = np.where(mask_arr == 6, 5, mask_arr)
    mask_arr = np.where(mask_arr == 3, 2, mask_arr)
    mask_arr = np.where(mask_arr == 0, 3, mask_arr)
    mask_arr = np.where(mask_arr == 7, 6, mask_arr)
    mask_arr = np.where(mask_arr == -1, 0, mask_arr)

    return mask_arr

for idx, point_path in enumerate(point_path_list[1:]):

    point_id = os.path.basename(point_path)
    print(f"{idx+1}: {point_id}")

    # if point_id != '20200401T140051_20200401T140048_T21LWF_1272S5614W':
    #     continue

    file_paths = [os.path.join(point_path, fname) for fname in os.listdir(point_path)]
    esri_label_path = [file for file in file_paths if file.endswith('ESRI.npy')][0]
    glc10_label_path = [file for file in file_paths if file.endswith('GLC10.npy')][0]
    dw_label_path = [file for file in file_paths if file.endswith('DW.npy')][0]
    esawc_label_path = [file for file in file_paths if file.endswith('ESAWC.npy')][0]
    
    try:
        esri_arr = np.load(esri_label_path)
    except ValueError:
        esri_label_path = [file for file in file_paths if file.endswith('ESRI.tif')][0]
        esri_arr = rioxarray.open_rasterio(esri_label_path).squeeze()
        esri_label_path = esri_label_path.split('.')[0] + '.npy'
    glc10_arr = np.load(glc10_label_path)
    dw_arr = np.load(dw_label_path)
    esawc_arr = np.load(esawc_label_path)

    esri_mask = remap_mask(esri_arr)
    np.save(esri_label_path, esri_mask)

    glc10_mask = remap_mask(glc10_arr)
    np.save(glc10_label_path, glc10_mask)

    dw_mask = remap_mask(dw_arr)
    np.save(dw_label_path, dw_mask)

    esawc_mask = remap_mask(esawc_arr)
    np.save(esawc_label_path, esawc_mask)


    if REMAP_HAND_LABEL:
        hand_path = [file for file in file_paths if file.endswith('hand.tif')][0]
        hand_arr = rioxarray.open_rasterio(hand_path)

        hand_mask = remap_mask(hand_arr.squeeze())

        hand_arr.values = np.expand_dims(hand_mask, 0)
        hand_arr.rio.to_raster(hand_path)

    s1_path = [file for file in file_paths if file.endswith('S1.npy')][0]
    s2_path = [file for file in file_paths if file.endswith('S2.npy')][0]
    meta_path = [file for file in file_paths if file.endswith('.csv')][0]

    s1_arr = np.load(s1_path)
    s2_arr = np.load(s2_path)
    meta_df = pd.read_csv(meta_path)

    plotter.plot_full_data(s1_arr, s2_arr, esri_mask, esawc_mask, dw_mask, glc10_mask, meta_df, True, point_id)


    




    
    

