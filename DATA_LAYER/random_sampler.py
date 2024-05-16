import argparse
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import xarray as xr
import pandas as pd
import random

from UTILS import utils

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Please provide a configuration ymal file for sampling points along HydroSHEDS river network.")
    parser.add_argument('--config_path',type=str, default='script/config.yml',help='Path to a configuration yaml file.' )
    
    args = parser.parse_args()
    config = utils.load_config(args.config_path)

    SAMPLE_METHOD = config['sample']['sample_mode']
    SAMPLE_LENGTH = config['sample']['sample_length']

    network_order_path = 'data/Amazon_HydroSHEDS_river_networks/Amazon_river_network_rasterized_by_order.tif'
    network_da_path = 'data/Amazon_HydroSHEDS_river_networks/Amazon_river_network_rasterized_by_drainage_area.tif'

    network_order = xr.open_dataarray(network_order_path).rename('riv_order')
    network_da = xr.open_dataarray(network_da_path).rename('upland_drainage_area')

    # Combine two array by coordinates
    network = xr.combine_by_coords([network_order, network_da])

    # Only keep data with not nan values 
    network_df = network.to_dataframe().dropna().reset_index().drop(['band', 'spatial_ref'], axis=1)

    # Export the full list of points of rasterized HydroSHEDS river networks
    # network_df.to_csv('Amazon_HydroSHEDS_river_networks/network_da_order_fulllist.csv')

    if SAMPLE_METHOD == 'RANDOM':
        # Create a list of index for random sampling 
        index_list = list(range(0, network_df.shape[0]))


        # Take 1000 random samples (points with coordinates, river order and upland drainage area) from the network_df
        sample_index = random.sample(index_list, SAMPLE_LENGTH)
        sample_df = network_df.iloc[sample_index]

        sample_df.to_csv(f'data/Amazon_HydroSHEDS_river_networks/network_points_{SAMPLE_LENGTH}_{SAMPLE_METHOD}.csv')

    elif SAMPLE_METHOD == 'STRATIFIED':
        # Define the number of samples
        num_samples = int(SAMPLE_LENGTH / 6)

        # Create an empty DataFrame to store the samples
        sample_df = pd.DataFrame()

        # Loop over each unique value in 'riv_order'
        for value in network_df['riv_order'].unique():
            # Sample the data
            temp_df = network_df[network_df['riv_order'] == value].sample(num_samples)
            # Append the samples to the samples DataFrame
            sample_df = pd.concat([sample_df, temp_df])
 
        sample_df.to_csv(f'data/Amazon_HydroSHEDS_river_networks/network_points_{SAMPLE_LENGTH}_{SAMPLE_METHOD}.csv')
