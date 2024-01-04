# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:54:40 2022

@author: s1135972
"""

#SHAPEFILE PREPPER

#This code splits multi-polygon shapefiles into lots of new shapefiles
#Specifically, one shapefile for every individual polygon in the input shapefile.
#It has been designed to specifically take HYDRObasins shapefiles as inputs (https://www.hydrosheds.org/downloads).
#The reason for doing this is to generate basin shapefiles suitable for the later clipping of DEM data using gdalwarp -cutline.
#This is part of a larger process to programmatically automate the extraction of global river networks using LSDTopoTools
#This script add the EPSG zone for UTM projection as an attribute, for ease of later projection.
#There is an option to add a buffer around margin of each of these shapefiles.

import geopandas as gpd
import math
import os

#Inputs
input_shapefile = 'Amazon_basin_v8.shp'    #Input shapefile: HydroBasin level 8 layer can get the full basin processed
input_shapefile_path = 'Amazon_HydroSHEDS_river_basin/'            #Path to input shapefile
output_files_path = 'basins/'   #Path for ouput shapefiles to be saved to
name_field = 'HYBAS_ID'                                                 #The name of a shapefile attribute column containing a unique identifier
area_field =  'SUB_AREA'                                                #The name of a shapefile attribute column containing the area of the polygon in square kilometres
buffer_radius = 0.01                                                    #The distance to buffer polgon edges. This is, irritatingly, in CRS units - degrees
basin_lower_thresh = 20     
basin_upper_thresh = 30000                                                #The threshold for basin area in square kilometres

#Function to return EPSG_code from lat/lon coordinates
def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code

#build some subdirectories for output files
path1 = os.path.join(output_files_path, 'shapefile_prepper_outputs', '')
if not os.path.exists(path1):
    os.mkdir(path1)
directories_to_create = ('shpfiles_for_subdivision','shpfiles_for_processing','shpfiles_main','')
for directory in directories_to_create:
    path2 = os.path.join(path1, directory)
    if not os.path.exists(path2):
        os.mkdir(path2)

#define subdirectories for outputs 
save_intermediate_shapefiles_to_path = '{}shpfiles_main/'.format(path1)
save_output_shapefiles_for_processing_to_path = '{}shpfiles_for_processing/'.format(path1)
save_output_shapefiles_for_subdivision_to_path = '{}shpfiles_for_subdivision/'.format(path1)

#Read in the input shapefile
basins = gpd.read_file('{}{}'.format(input_shapefile_path,input_shapefile), crs="EPSG:4326")

#Filter basins based area thresholds 
filtered_basins = basins[(basins[area_field]>basin_lower_thresh) & (basins[area_field]<basin_upper_thresh)]

#Buffer shape edges
#Shapefile must be re-projected to meters such as epsg: 3857
buffered_basins = filtered_basins.geometry.buffer(buffer_radius, resolution=2)
buffered_basins.to_file('{}basins_for_processing.shp'.format(save_intermediate_shapefiles_to_path))

#Filter out basins too large to process and save as new shapefile
basins_for_subdivision = basins[basins[area_field]>basin_upper_thresh]
basins_for_subdivision.to_file('{}basins_for_subdivision.shp'.format(save_intermediate_shapefiles_to_path))



for shp, path in zip(('basins_for_processing.shp','basins_for_subdivision.shp'),(save_output_shapefiles_for_processing_to_path,save_output_shapefiles_for_subdivision_to_path)):
    basins2 = gpd.read_file('{}{}'.format(save_intermediate_shapefiles_to_path,shp), crs="EPSG:4326")
    
    #iterate through shapefile polygons, saving each individually projected in relevant UTM zone:
    for index, row in basins2.iterrows():
        #Access each polygon as separate geopandas df:
        gpdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(row.geometry))
        
        #Find centroid and corresponding UTM zone:
        lon = gpdf.centroid.x.values
        lat = gpdf.centroid.y.values
        EPSG = convert_wgs_to_utm(lon,lat)
        gpdf['EPSG'] = EPSG
        print (gpdf)
        
        #save each polygon individually
        #gpdf.set_crs(crs='EPSG:{}'.format(EPSG)).to_file('{}{}.shp'.format(save_shapefiles_to_path,basins[name_field].iloc[index]))
        gpdf.to_file('{}{}.shp'.format(path,basins[name_field].iloc[index]))
        
        #read out the EPSG for every shapefile
        basin = gpd.read_file('{}{}.shp'.format(path,basins[name_field].iloc[index]))
        EPSG2 = int(basin['EPSG'].values)
        print ('EPSG2 is {}'.format(EPSG2))