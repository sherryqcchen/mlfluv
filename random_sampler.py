import xarray as xr
import numpy as np
import random

network_order_path = 'Amazon_HydroSHEDS_river_networks/Amazon_river_network_rasterized_by_order.tif'
network_da_path = 'Amazon_HydroSHEDS_river_networks/Amazon_river_network_rasterized_by_drainage_area.tif'

network_order = xr.open_dataarray(network_order_path).rename('riv_order')
network_da = xr.open_dataarray(network_da_path).rename('upland_drainage_area')

# Combine two array by coordinates
network = xr.combine_by_coords([network_order, network_da])

# Only keep data with not nan values 
network_df = network.to_dataframe().dropna().reset_index().drop(['band', 'spatial_ref'], axis=1)

# Create a list of index for random sampling 
index_list = list(range(0, network_df.shape[0]))

# Take 1000 random samples (points with coordinates, river order and upland drainage area) from the network_df
SAMPLE_LENGTH = 1000
sample_index = random.sample(index_list, SAMPLE_LENGTH)
sample_df = network_df.iloc[sample_index]

sample_df.to_csv('Amazon_HydroSHEDS_river_networks/network_da_order_sample.csv')