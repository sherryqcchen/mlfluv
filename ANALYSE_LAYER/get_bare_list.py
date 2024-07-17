import os

import pandas as pd

def find_csv_file(path):
    """Find the first CSV file in the given directory path."""
    for file in os.listdir(path):
        if file.endswith('.csv'):
            return os.path.join(path, file)
    return None

def read_and_rename_columns(csv_file):
    """Read specific columns from the CSV file and rename them."""
    df = pd.read_csv(csv_file)
    # Check if required columns exist in the DataFrame
    required_columns = ['point', 'riv_order', 'drainage_area','acc_rainfall','cloud_prob']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"'{column}' column not found in {csv_file}")

    # Create a new DataFrame with renamed columns
    new_df = df[['point', 'riv_order', 'drainage_area','acc_rainfall','cloud_prob']].copy()
    new_df.columns = ['coordinates', 'riv_order', 'upland_drainage_area','acc_rainfall','cloud_prob']
    return new_df



def get_one_LULC_data_list(lulc='bare'):
    if lulc=='init_dataset':
        clean_data_path = 'data/clean_data/mlfluv_s12lulc_data_water_from_STRATIFIED_6000'
    else:
        clean_data_path = f'data/clean_data/mlfluv_incremental_data_{lulc}'
    bare_folder_paths = [os.path.join(clean_data_path, file) for file in os.listdir(clean_data_path)]
    print(f"{lulc} data amount: {len(bare_folder_paths)}")

    # print(bare_folder_paths)
    all_dfs = []
    for folder in bare_folder_paths:
        csv_path = find_csv_file(folder)
        if csv_path:
            df = read_and_rename_columns(csv_file=csv_path)
            all_dfs.append(df)
        else:
            print(f"No CSV file found in the specified path: {folder}")

    if all_dfs:
        big_df = pd.concat(all_dfs, ignore_index=True)
        out_csv_path = f'data/train_meta/network_points_{lulc}.csv'
        big_df.to_csv(out_csv_path, index=False)
        # print(big_df)

    else:
        print("No valid CSV files found in any of the specified paths.")
            
get_one_LULC_data_list('bare')
get_one_LULC_data_list('sediment')
get_one_LULC_data_list('urban')
get_one_LULC_data_list('init_dataset')