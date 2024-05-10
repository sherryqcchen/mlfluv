import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import csv
import requests
import io
import ee
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import NearestNDInterpolator
from gee_s1_ard.python_api import wrapper as wp
from UTILS import plotter

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

def get_lat_lng_string(point_coords):
    """
    Convert a coordinate of a point in EPSG:4326 to string which can be used for naming data.
    The floats are round to keep 2 demicals and then multiplied by 100 to remove "." in the number. 

    Args:
        point_coords (list): [lng, lat], lng and lat are floats.

    Returns:
        coord_str (string): an example "3467N11702W" means [-117.02, 34.67] in [lng, lat].
    """

    lng = point_coords[0]
    lat = point_coords[1]
    if lng<0:
        lng_str = str(int(round(abs(lng), 2)*100)) + "W"
    else:
        lng_str = str(int(round(abs(lng), 2)*100)) + "E" 

    if lat<0:
        lat_str = str(int(round(abs(lat), 2)*100)) + "S"
    else:
        lat_str = str(int(round(abs(lat), 2)*100)) + "N" 

    coord_str = lat_str + lng_str

    return coord_str

def ee_buffer_point(coordinates):
    """
    An Earth Engine Python API functiond. 
    Buffer a point to a rectangle with size approximating to 2048*2048 using ee.Geometry object
    
    Args:
        coordinates (list): [lng, lat]

    Returns:
        aoi (ee.Geometry): the buffered rectangle geometry
    """
    point = ee.Geometry.Point(coordinates)
    aoi = point.buffer(distance=2560, maxError=4).bounds()
    # print(aoi.getInfo())
    # buffered_area = aoi.area(maxError=1).getInfo()
    # print(buffered_area)
    # print(f"Target area: {2048*2048}")
    return aoi

def ee_mask_invalid_values(image, low_threshold=-1, up_threshold=9):
    mask = image.gt(low_threshold).And(image.lt(up_threshold))
    masked = image.updateMask(mask)
    return masked
    
def ee_count_masked_dw_percent(image, geometry):
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
    
    image_pixels = ee_mask_invalid_values(ee.Image(image)).select('label').reduceRegion(reducer=ee.Reducer.count(), geometry=geometry, scale=image.select('label').projection().nominalScale())
    # Calculate the pixel count 
    total_pixels = ee.Image(1).reduceRegion(reducer= ee.Reducer.count(), geometry=geometry, scale=image.select('label').projection().nominalScale())

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
        to_list =   [1,  2,  2,  2,  4,  5,  0,  3,  2,  2,  0]
        band_name = 'Map'

    elif lulc_type == 'dynamic_world':
        from_list = [0, 1, 2, 3, 5, 4, 6, 7, 8]
        to_list =   [3, 1, 2, 2, 2, 2, 4, 5, 0]
        band_name = 'label'
    
    elif lulc_type == 'from_glc10':
        from_list = [10, 20, 30, 40, 60, 80, 90, 100]
        to_list =   [ 2,  1,  2,  2,  3,  4,  5,  0]
        band_name = 'b1'
    
    elif lulc_type == 'esri_land_cover':
        from_list = [1, 2, 4, 5, 7, 8, 9, 11]
        to_list =   [3, 1, 2, 2, 4, 5, 0, 2]
        band_name = 'b1'

    else:
        print('Unrecognised LULC map is given.')

    image_remap = image.remap(from_list, to_list, defaultValue=-999, bandName=band_name)

    return image_remap

def convert_ee_image_to_np_arr(img, band_names, geometry):
    """
    Resample and then convert ee images from ee.Image object to .npy file.
    Default coordinate system: epsg:4326, spatial resolution: 10 meters.

    Args:
        img(ee.Image object): Sentinel-1/2 image or land cover map, data type ee.Image.
        band_names(string or list of strings): band names of the ee.Image objected.
        geometry(ee.Geometry object): bounds of downloaded data.
 
    Returns:
        img_arr: converted ee image, data type ndarray, shape (514, 514, band number)
    """

    img_resample = img.select(band_names).reproject(crs='EPSG:4326', scale=10)
    img_projection = img_resample.select(0).projection()
    img_url = img_resample.toFloat().getDownloadURL({'name': 's1_image',
                                                   'bands': band_names,
                                                   'region': geometry,
                                                   'scale': 10,
                                                   'format': 'NPY'})
    img_response = requests.get(img_url)
    img_data = np.load(io.BytesIO(img_response.content))
    # reshape data: unpack numpy array of tuples into ndarray
    # Method from: https://stackoverflow.com/questions/55852450/how-do-i-unpack-a-numpy-array-of-tuples-into-an-ndarray
    img_arr = rf.structured_to_unstructured(img_data)

    return img_arr, img_projection

def interpolator(data):
    """
    Interpolate missing data for the connecting areas between tiles
    Usually Sentinel-1 images would have -inf values in the data array after running resampling in the function convert_ee_image_to_np_arr()
    Args:
        data: data array
    Returns:
        data array
    """
    if np.all(np.isfinite(data)) is False:
        mask = np.where(np.isfinite(data))
        interp = NearestNDInterpolator(np.transpose(mask), data[mask])
        data = interp(*np.indices(data.shape))
    return data


def download_1_point_data(coords, river_order, drainage_area,  year=2020, VIS_OPTION=False, TIF_option=False):
    """
    Filter Sentinel-1/2 images and 4 LULC labels for a given point in a given year.
    Download all the images as ".npy" files and saved under the directory of point_id

    Args:
        year (int, optional): The year to filter data. Defaults to 2020.
        coords (list, optional): A center point which will be buffered to a square later to download data. 
        river_order (int): river order obtained from HydroSHEDS dataset.
        drainage_area (float): upland drainage area of the chosen point in the entire basin, data obtained from HydroSHEDS dataset.

    Returns:
        no return.
    """    
    coord_string = get_lat_lng_string(coords)

    aoi = ee_buffer_point(coords)
    # print(river_order)

    # Filter from the first day to the last day of a given year 
    start_date = ee.Date.fromYMD(year,1, 1)
    end_date = start_date.advance(1, 'year')

    col_filter = ee.Filter.And(ee.Filter.bounds(aoi),ee.Filter.date(start_date, end_date))

    col_filter_for_sentinel = ee.Filter.And(ee.Filter.bounds(aoi),ee.Filter.date(ee.Date.fromYMD(year,9, 1), ee.Date.fromYMD(year,11, 1)))

    # Filter image collection from Google Dynamic World dataset, sentinel-1, sentinel-2
    dw_col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filter(col_filter_for_sentinel)
    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD').filter(col_filter_for_sentinel)
    s2_col = ee.ImageCollection('COPERNICUS/S2').filter(col_filter_for_sentinel)
    print(dw_col.size().getInfo())

        # DW is made on top of Sentinel-2, so they have the same system_index property and these two collections can be joined together 
    dws2_col = ee.Join.saveFirst('s2_img').apply(dw_col, s2_col, ee.Filter.equals(leftField='system:index', rightField='system:index'))

        # Define a function to count masked pixels in each DW label
    dws2_col = ee.ImageCollection(dws2_col).map(lambda image: ee_count_masked_dw_percent(image, aoi))
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

        # get s2 image system index 
    s2_id = s2_image.get('system:index').getInfo()
        # print('S2 image of clear date:', s2_image.getInfo())

        # add a property to sentinel-1 collection to find the closest date to the clear data
    def ee_get_date_diff(image):
        return image.set('date_diff', 
                            ee.Number(image.get('system:time_start')).subtract(clear_s2_date.millis()).abs())
        
    s1_col = s1_col.map(ee_get_date_diff).sort('date_diff')

        # The first image in the sorted collection is the image of the closest date
        # we use mosaic here because sometime it only has half a tile, in this case we mosaic the two tiles with closest date
        # s1_image = s1_col.mosaic().select('VV', 'VH') 
    clear_s1_date = ee.Date(s1_col.first().get('system:time_start'))
        
    print(f"The chosen s1 image date: {clear_s1_date.format('Y-MM-dd').getInfo()}")
        
        # preprocess s1 image using https://github.com/adugnag/gee_s1_ard/blob/main/python-api/s1_ard.py
        # Parameters
    dem_cop = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').filterBounds(aoi).mosaic()
    parameter = {'START_DATE': clear_s1_date.advance(-1, 'day'),
                    'STOP_DATE': clear_s1_date.advance(1, 'day'),        
                    'POLARIZATION': 'VVVH',
                    'ORBIT' : None,
                    'ORBIT_NUM': None,
                    'ROI': aoi,
                    'APPLY_BORDER_NOISE_CORRECTION': False,
                    'APPLY_SPECKLE_FILTERING': True,
                    'SPECKLE_FILTER_FRAMEWORK':'MULTI',
                    'SPECKLE_FILTER': 'GAMMA MAP',
                    'SPECKLE_FILTER_KERNEL_SIZE': 9,
                    'SPECKLE_FILTER_NR_OF_IMAGES':10,
                    'APPLY_TERRAIN_FLATTENING': True,
                    'DEM': dem_cop,
                    'TERRAIN_FLATTENING_MODEL': 'VOLUME',
                    'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER':0,
                    'FORMAT': 'DB',
                    'CLIP_TO_ROI': True,
                    'SAVE_ASSET': False,
                    'ASSET_ID': "users/qiuyangschen"
                    }
        # pre-process s1 collection
    s1_processed = wp.s1_preproc(parameter)
    if s1_processed.size().getInfo() != 0:
        s1_id = s1_processed.first().get('system:index')
        s1_image = s1_processed.mosaic().set('system:index', s1_id)#.select('VV', 'VH')

        # Convert s1_image, s2_image to numpy arrays
        if all_exist(S1_BANDS, s1_image.bandNames().getInfo()) and all_exist(S2_BANDS, s2_image.bandNames().getInfo()):
            s1_data, s1_proj = convert_ee_image_to_np_arr(s1_image, S1_BANDS, aoi)
            s2_data, s2_proj = convert_ee_image_to_np_arr(s2_image, S2_BANDS, aoi)

            # Interpolate missing data in s1, s2 images
            s1_filled_data = interpolator(s1_data)
            s2_filled_data = interpolator(s2_data)

            # Visualize s1, s2 images: plot only works in vscode interactive window 
            if VIS_OPTION:
                plotter.plot_s1(s1_filled_data, 'vv')
                plotter.plot_s2_rgb(s2_filled_data)
                
            # Merge classes in different LULC products
            esri_label = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS").filter(col_filter).mosaic()
            esri_remap = remap_lulc(esri_label, 'esri_land_cover')
                
            esawc_label = ee.Image('ESA/WorldCover/v100/2020')
            esawc_remap = remap_lulc(esawc_label, 'esa_world_cover')   
                
            glc10_label = ee.ImageCollection("projects/sat-io/open-datasets/FROM-GLC10").mosaic()
            glc10_remap = remap_lulc(glc10_label, 'from_glc10')
                
            dw_remap = remap_lulc(dws2_image, 'dynamic_world')  

                # Read these remapped tiles into numpy array, there remapped bands are all called "remapped".
            esri_arr, _ = convert_ee_image_to_np_arr(esri_remap, 'remapped', aoi)
            esawc_arr, _ = convert_ee_image_to_np_arr(esawc_remap, 'remapped', aoi)
            glc10_arr, _ = convert_ee_image_to_np_arr(glc10_remap, 'remapped', aoi)
            dw_arr, _ = convert_ee_image_to_np_arr(dw_remap, 'remapped', aoi)

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
                
            if accumulative_rainfall_total.size().getInfo() == 0:
                rainfall_sum = ee.Number(-999)
                print('This point has no available rainfall data.')
            else:
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

                # Exporting data, we use the system index of Sentinel-2 and DW data (they share the same index) as part of the point_id
                # In some cases if the sampling points are close to each other and occured on the same Sentinel-2 tile
                # We also use lng, lat info in the point_id 
            point_id = s2_id + "_" + coord_string

            data_path = '/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_s12lulc_data_STRATIFIED'
            # data_path = '/exports/csce/datastore/geos/users/s2135982/MLFLUV_DATA/data_mining_bareground_samples'
            point_path = os.path.join(data_path, point_id) 

            if not os.path.exists(point_path):
                os.makedirs(point_path)
                
            meta_fname = os.path.join(point_path, point_id + "_meta.csv")
                
                # Save all the meta info in a dictionary
            meta_dict = {}
            meta_dict['point'] = coords
            meta_dict['year'] = year
            meta_dict['riv_order'] = river_order
            meta_dict['drainage_area'] = drainage_area
            meta_dict['aoi'] = aoi.coordinates().getInfo()
            meta_dict['projection'] = s2_proj.getInfo()
            meta_dict['s1_id'] = s1_image.get('system:index').getInfo()
            meta_dict['s2_id'] = s2_image.get('system:index').getInfo()
            meta_dict['s1_date'] = clear_s1_date.format('Y-MM-dd').getInfo()
            meta_dict['s2_date'] = clear_s2_date.format('Y-MM-dd').getInfo()
            meta_dict['DW_cover_percentage'] = max_percent.getInfo()
            meta_dict['acc_rainfall'] = rainfall_sum.getInfo()
            meta_dict['cloud_prob'] = s2_cloud_prob.getInfo()

            with open(meta_fname, 'w') as f:
                w = csv.DictWriter(f, meta_dict.keys())
                w.writeheader()
                w.writerow(meta_dict)

            np.save(os.path.join(point_path, point_id+'_S1.npy'), s1_filled_data)
            np.save(os.path.join(point_path, point_id+'_S2.npy'), s2_filled_data)
            np.save(os.path.join(point_path, point_id+'_ESRI.npy'), esri_arr)
            np.save(os.path.join(point_path, point_id+'_ESAWC.npy'), esawc_arr)
            np.save(os.path.join(point_path, point_id+'_DW.npy'), dw_arr)
            np.save(os.path.join(point_path, point_id+'_GLC10.npy'), glc10_arr)

            # if TIF_option:
                

        else:
            print('S1 or S2 has missing bands, skip this point.')
            pass
    else:
        print('Sentinel-1 data missing, skip this point.')
        pass


        
if __name__ == "__main__":

    service_account = 'earthi-ubuntu@sen12flood-qiuyangchen.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, './sen12flood-qiuyangchen-8fdb42008616.json')

    ee.Initialize(credentials)

    # # If you don't have Earth Engine Service Account Credentials, using following lines
    # ee.Authenticate()
    # ee.Initialize()

    # print(os.getcwd())
    year = 2020

    # Buffer the point to a rectangle called aoi with 2048*2048 size
    # points_path = Path('/exports/csce/datastore/geos/users/s2135982/rivertools/mlfluv/Amazon_HydroSHEDS_river_networks/network_sediment_rich_sample.csv')
    # points_path = Path('/exports/csce/datastore/geos/users/s2135982/rivertools/mlfluv/Amazon_HydroSHEDS_river_networks/network_mining_bareground_sample.csv')
    points_path = Path('/exports/csce/datastore/geos/users/s2135982/rivertools/mlfluv/Amazon_HydroSHEDS_river_networks/network_da_order_sample_6000_STRATIFIED.csv')
    
    point_list = []

    # Specify the encoding if needed (e.g., utf-8, latin-1, utf-16)
    # encoding = 'utf-16'
    with open(points_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            coord = tuple([float(row['x']), float(row['y'])])
            # coord = tuple(float(x.strip('\'"')) for x in row['coordinates'].strip('()').split(','))
            riv_order = int(float(row['riv_order']))
            da = float(row['upland_drainage_area'])
            point_list.append((coord,riv_order, da))

    # print(point_list)
    for idx, (point_coord, riv_ord, da) in enumerate(point_list[3744:]):

        print(f"{idx+3744}: {point_coord}")
        try:
            download_1_point_data(point_coord, riv_ord, da, VIS_OPTION=False)
        except ee.ee_exception.EEException as err:
            print("An EEException occurred:", err)
            print('No valid clear data found within the given time range')
            continue
 
