import os
import datetime
import json
import requests
import io
import ee
import cv2
import geemap
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import NearestNDInterpolator

S1_BANDS = ['VV', 'VH']
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

def all_exist(full_list, given_list):
    """
    A function to check whether all  required bands are in filtered ee.Image objects
    Args:
        full_list: required list of band names
        given_list: given list of band names

    Returns: True or False, if True, all the required bands exist in the given bands
    """
    return all(any(i in j for j in given_list) for i in full_list)

def ee_buffer_point(coordinates):
    """
    An Earth Engine Python API functiond. 
    Buffer a point to a rectangle with size approximating to 2048*2048 using ee.Geometry object
    
    Args:
        coordinates (list): [lng, lat]

    Returns:
        aoi (ee.Geometry): the buffered rectangle geometry
    """
    point = ee.Geometry.Point(coords)
    aoi = point.buffer(distance=1024, maxError=4).bounds()
    # print(aoi.getInfo())
    # buffered_area = aoi.area(maxError=1).getInfo()
    # print(buffered_area)
    # print(f"Target area: {2048*2048}")
    return aoi

def ee_count_masked_dw_percent(image):
    """
    Count valid (not NaN) pixel percentage in a Dynamic World (DW) label. 
    The cloudy pixels in a DW label is usually masked out as NaN or null.

    Args:
        image (ee.Image): one image in the DW image collection

    Return:    
        image (ee.Image): with extra property 'maked_percentage' and 'date'
    """        
    # image = ee.Image(image)
    
    # Calculate the pixel count in a DW label
    image_pixels = ee.Image(image).select('label').reduceRegion(reducer=ee.Reducer.count(), geometry=aoi, scale=image.select('label').projection().nominalScale())
    # Calculate the pixel count 
    total_pixels = ee.Image(1).reduceRegion(reducer= ee.Reducer.count(), geometry=aoi, scale=image.select('label').projection().nominalScale())

    masked_percent = ee.Number(image_pixels.get('label')).divide(total_pixels.get('constant'))

    return image.set('masked_percentage', masked_percent).set('date', image.date().format('YYYY-MM-dd'))

def remap_lulc(image, lulc_type='esa_world_cover'):
    """
    Remap the class values of three land use and land cover (LULC) products to MLFluv classes
    In the MLFluv, water: 0, tree:1, shallow-rooted vegetation: 2, crops:3, build-up:4, 
    fluvial sediment:5, bareland: 6, ice/snow:7, clouds are set to -999. 

    Args:
        image (ee.Image): land cover map, with one band in the image saving class values in a list of integers.
        lulc_type (str, optional): key to indicate which LULC map is used. Defaults to 'esa_world_cover'. Choose from 'esa_world_cover', 'dynamic_world', 'from_glc10', 'esri_land_cover'

    Returns:
        image_remap (ee.Image): remapped land cover map.
    """        
    
    if lulc_type == 'esa_world_cover':
        from_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
        to_list =   [1,  2,  2,  3,  4,  6,  7,  0,  2,  2,  7]
        band_name = 'Map'

    elif lulc_type == 'dynamic_world':
        from_list = [0, 1, 2, 3, 5, 4, 6, 7, 8]
        to_list =   [0, 1, 2, 2, 2, 3, 4, 6, 7]
        band_name = 'label'
    
    elif lulc_type == 'from_glc10':
        from_list = [10, 20, 30, 40, 60, 80, 90, 100]
        to_list =   [ 3,  1,  2,  2,  0,  4,  6,  7]
        band_name = 'b1'
    
    elif lulc_type == 'esri_land_cover':
        from_list = [1, 2, 4, 5, 7, 8, 9, 11]
        to_list =   [0, 1, 2, 3, 4, 6, 7, 2]
        band_name = 'b1'

    else:
        print('Unrecognised LULC map is given.')

    image_remap = image.remap(from_list, to_list, defaultValue=-999, bandName=band_name)

    return image_remap


if __name__ == "__main__":

    service_account = 'earthi-ubuntu@sen12flood-qiuyangchen.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, './sen12flood-qiuyangchen-8fdb42008616.json')

    ee.Initialize(credentials)

    # If you don't have Earth Engine Service Account Credentials, using following lines
    # ee.Authenticate()
    # ee.Initialize()

    print(os.getcwd())

    # Buffer the point to a rectangle called aoi with 2048*2048 size
    coords = [-49.685515651616484,-2.665628314244678]
    aoi = ee_buffer_point(coords)
    
    year = 2020

    # Filter from the first day to the last day of a given year 
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, 'year')

    col_filter = ee.Filter.And(ee.Filter.bounds(aoi),ee.Filter.date(start_date, end_date))

    # Filter image collection from Google Dynamic World dataset, sentinel-1, sentinel-2
    dw_col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filter(col_filter)
    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD').filter(col_filter)
    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filter(col_filter)
    print(dw_col.size().getInfo())

    # DW is made on top of Sentinel-2, so they have the same system_index property and these two collections can be joined together 
    dws2_col = ee.Join.saveFirst('s2_img').apply(dw_col, s2_col, ee.Filter.equals(leftField='system:index', rightField='system:index'))

    # Define a function to count masked pixels in each DW label
    dws2_col = ee.ImageCollection(dws2_col).map(ee_count_masked_dw_percent)
    max_percent = dws2_col.aggregate_max('masked_percentage')
    print(max_percent.getInfo())

    # Get an array of masked_percentage and an array of date for all images in the DW collection
    percent_arr = ee.Array(dws2_col.aggregate_array('masked_percentage'))
    date_arr = ee.Array(dws2_col.aggregate_array('system:time_start'))

    # get the first max percent image index
    max_idx = percent_arr.argmax()
    max_time_start = date_arr.get(max_idx)
    # print(dw_col.aggregate_array('system:time_start').getInfo())
    print(f"I find the first date of DW lable with max valid pixel percentage: {ee.Date(max_time_start).format('Y-MM-dd').getInfo()}")
    
    # Get this first clear date
    clear_s2_date = ee.Date(max_time_start)

    # Get DW label of this date
    dws2_image = ee.Image(dws2_col.filterDate(clear_s2_date, clear_s2_date.advance(1, 'day')).first())

    s2_image = ee.Image(dws2_image.get('s2_img')).select('B.*')
    # print('S2 image of clear date:', s2_image.getInfo())

    # add a property to sentinel-1 collection to find the closest date to the clear data
    def ee_get_date_diff(image):

        return image.set('date_diff', 
                         ee.Number(image.get('system:time_start')).subtract(clear_s2_date.millis()).abs())
    
    s1_col = s1_col.map(ee_get_date_diff).sort('date_diff')

    # The first image in the sorted collection is the image of the closest date
    # we use mosaic here because sometime it only has half a tile, in this case we mosaic the two tiles with closest date
    s1_image = s1_col.mosaic().select('VV', 'VH') 
    clear_s1_date = ee.Date(s1_col.first().get('system:time_start'))
    
    print(f"The chosen s1 image date: {clear_s1_date.format('Y-MM-dd').getInfo()}")
    
    # TODO: preprocess s1 image using https://github.com/adugnag/gee_s1_ard/blob/main/python-api/s1_ard.py

    # Merge classes in different LULC products
    esri_label = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS").filter(col_filter).mosaic()
    esri_remap = remap_lulc(esri_label, 'esri_land_cover')
    
    esawc_label = ee.Image('ESA/WorldCover/v100/2020')
    esawc_remap = remap_lulc(esawc_label, 'esa_world_cover')   
    
    glc10_label = ee.ImageCollection("projects/sat-io/open-datasets/FROM-GLC10").mosaic()
    glc10_remap = remap_lulc(glc10_label, 'from_glc10')
    
    dws2_remap = remap_lulc(dws2_image, 'dynamic_world')  

    # Get accumulative rainfall from CHIRPS dataset
    rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')\
                    .select('precipitation')\
                    .filter(ee.Filter.Or(
                            ee.Filter.date(clear_s1_date, clear_s2_date),
                            ee.Filter.date(clear_s2_date, clear_s1_date)))\
                    .filterBounds(aoi)
    
    # Accumulative rainfall between clear_s1_date and clear_s2_date, 
    accumulative_rainfall_total = rainfall.reduce(ee.Reducer.sum())\
                                    .reduceRegion(reducer= ee.Reducer.mean(), geometry=aoi, scale=5000)
    
    rainfall_sum = accumulative_rainfall_total.get('precipitation_sum') # the unit is mm
    print(f"The average accumulative rainfall within aoi is {rainfall_sum.getInfo()} mm.")

    # Get S2 cloud probability for the s2 image
    s2_cloud_prob= ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")\
                    .filter(col_filter)\
                    .filterDate(clear_s2_date.getRange('day'))\
                    .first()\
                    .select('probability')\
                    .reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=5000)\
                    .get('probability')
    print(f"The cloud probability is:{s2_cloud_prob.getInfo()}")
                    
