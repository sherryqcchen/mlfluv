import os
import datetime
import json
import requests
import io
import ee
import cv2
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

    # Filter image collection from Google Dynamic World dataset
    dw_col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filter(col_filter)
    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filter(col_filter)
    print(dw_col.size().getInfo())

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
    clear_date = ee.Date(max_time_start)

    # Get DW label of this date
    dws2_image = ee.Image(dws2_col.filterDate(clear_date, clear_date.advance(1, 'day')).first())

    s2_image = ee.Image(dws2_image.get('s2_img')).select('B.*')
    # print('Sentinel-2 image of clear date:', s2_image.getInfo())

    clear_range_end_date = clear_date.advance(2, 'week')

    print(clear_range_end_date.format('Y-MM-dd').getInfo())




    




