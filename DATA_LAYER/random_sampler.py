import xarray as xr
import pandas as pd
import random

SAMPLE_METHOD = 'RANDOM' #  'STRATIFIED'


network_order_path = 'Amazon_HydroSHEDS_river_networks/Amazon_river_network_rasterized_by_order.tif'
network_da_path = 'Amazon_HydroSHEDS_river_networks/Amazon_river_network_rasterized_by_drainage_area.tif'

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
    sample_length = 12000
    sample_index = random.sample(index_list, sample_length)
    sample_df = network_df.iloc[sample_index]

    sample_df.to_csv(f'Amazon_HydroSHEDS_river_networks/network_da_order_sample_{sample_length}_{SAMPLE_METHOD}.csv')

elif SAMPLE_METHOD == 'STRATIFIED':
    # Define the number of samples
    num_samples = 2000

    # Create an empty DataFrame to store the samples
    sample_df = pd.DataFrame()

    # Loop over each unique value in 'riv_order'
    for value in network_df['riv_order'].unique():
        # Sample the data
        temp_df = network_df[network_df['riv_order'] == value].sample(num_samples)
        # Append the samples to the samples DataFrame
        sample_df = pd.concat([sample_df, temp_df])
    sample_length = num_samples * 6    
    sample_df.to_csv(f'Amazon_HydroSHEDS_river_networks/network_da_order_sample_{sample_length}_{SAMPLE_METHOD}.csv')
