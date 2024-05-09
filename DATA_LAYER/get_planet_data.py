'''

Author: Qiuyang Chen
Contact: qiuyangschen@gmail.com

This scrip is used to download validating Planet Scene images for MLFluv. 
It uses AOI geometry  and clear s2 date from each data point in MLFluv to filter data.
It fillter the scenes with less than 5% cloud cover with a month of the given date.
Once a qualified scene is found, the scrip creates the order in batches and download orders once the ordered are ready. 
Quite often the Planet Order API takes time to run the oder jobs after the order creation. 
This script should be run (at least) twice: to create order for the first time, and later run again to download orders. 
It is suggested to monitor order status on the Planet Account website. 
'''

import datetime
import json
import os
from urllib.request import HTTPBasicAuthHandler

from matplotlib import pyplot as plt
import pandas as pd
import requests
import asyncio

import planet
from planet import Auth, Session, DataClient, OrdersClient, order_request, reporting

from label_prepare import get_bounding_box
from rasterio.transform import from_origin


# Helper function to printformatted JSON using the json module
def p(data):
    print(json.dumps(data, indent=2))

def items_to_scenes(items):
    # Method from:# https://github.com/planetlabs/notebooks/blob/6cc220ff6db246353af4798be219ee1fe7e858b0/jupyter-notebooks/crossovers/ps_l8_crossovers.ipynb
    # A function to convert planet searched item list to scenes
    item_types = []

    def _get_props(item):
        props = item['properties']
        props.update({
            'thumbnail': item['_links']['thumbnail'],
            'item_type': item['properties']['item_type'],
            'id': item['id'],
            'acquired': item['properties']['acquired'],
            'cloud_cover': item['properties']['cloud_cover'],
            'cloud_percent': item['properties']['cloud_percent'],
            'footprint': item['geometry']
        })
        return props  
    
    scenes = pd.DataFrame(data=[_get_props(i) for i in items])
    # acquired column to index, it is unique and will be used a lot for processing
    scenes.index = pd.to_datetime(scenes['acquired'])
    del scenes['acquired']
    scenes.sort_index(inplace=True)

    return scenes

def display_thumbnails(scenes, limit=5):
    # A function to display thumbnails of scenes in a list
    from PIL import Image
    from IPython.display import display
    from io import BytesIO
    for thumb_url in scenes['thumbnail'].tolist()[:limit]:
        print(thumb_url)
        r = requests.get(thumb_url, auth=(API_KEY, ""))
        i = Image.open(BytesIO(r.content))
        plt.imshow(i)
        plt.show()
        # display(i)

async def search_Planet(name, filter, item):
    async with Session() as sess:
        cl = DataClient(sess)
        results = cl.search(name=name, search_filter=filter, item_types=item)
        list = [i async for i in results]
        return list     
    
async def create_PS_order(request):
    # Method from https://planet-sdk-for-python-v2.readthedocs.io/en/latest/python/sdk-guide/#your-orders-client
    async with Session() as sess:
        cl = OrdersClient(sess)
        with reporting.StateBar(state='creating') as bar:
            # create order
            order = await cl.create_order(request)
            bar.update(state='created', order_id=order['id'])
            # poll
        #     await cl.wait(order['id'], callback=bar.update_state)
        # # download
        # await cl.download_order(order_id=order['id'], directory=dir)

async def list_PS_order():
    # Method from https://planet-sdk-for-python-v2.readthedocs.io/en/latest/python/sdk-guide/#collecting-results
    async with Session() as sess:
        client = OrdersClient(sess)
        # orders_list = collect(client.list_orders())
        orders_list = [o async for o in client.list_orders()]
        return orders_list
    
async def download_order(order_id, dir):
    async with Session() as sess:
        client = OrdersClient(sess)
        order = await client.get_order(order_id)
        # Download order
        await client.download_order(order['id'], directory=dir)
        # return order

async def cancel_order(order_id):
    async with Session() as sess:
        client = OrdersClient(sess)
        await client.cancel_order(order_id=order_id)


def search_planet_data(job_name, geometry, date_start, date_end):
    """
    Search available planet data using Data API by given polygon geometry, start date and end date.

    Args:
        geometry (list): list of coordinates of the polygon
        date_start (string): date in the format 'YYYY-MM-DD'
        date_end (string): date in the format 'YYYY-MM-DD'
    Returns:
        list: a list of links to access filtered Planet data 
    """            

    # Create filters for Data API search
    geometry_filter = {"type": "GeometryFilter", "field_name": "geometry", "config": geometry}
    date_range_filter = {"type": "DateRangeFilter", "field_name": "acquired", "config": 
                        {"gte": date_start.isoformat()+'T00:00:00.000Z',
                        "lt": date_end.isoformat()+'T00:00:00.000Z'}}
    cloud_cover_filter = {"type": "RangeFilter", "field_name": "cloud_cover", "config": 
                        {"lte": 0.05}} # Cloud cover <= 5%
    
    # combine our geo, date, cloud filters
    combined_filter = {"type": "AndFilter", "config": [geometry_filter, date_range_filter, cloud_cover_filter]}

    # searching items and assest
    item_type = ["PSScene"]

    # Run a quick search for our data
    result_list = asyncio.run(search_Planet(job_name, combined_filter, item_type))
    # print(len(result_list))
    return result_list


def monitor_Planet_account_quota(API_KEY):
    # A function using Planet experimental API to manage subscription quota
    # Note: this quota shown from the PlanetLab user account on the website is not updated until 24 hours after requesting new data
    subscription_url = "https://api.planet.com/auth/v1/experimental/public/my/subscriptions"
    subscription_res = requests.get(subscription_url, auth=(API_KEY, '')).json()
    rest_quota = subscription_res[0]['quota_sqkm'] - subscription_res[0]['quota_used']

    print(f"The left quota is : {rest_quota} km2.")


def order_100_data(path_list):
    """
    Create and download maximum 100 Planet jobs from a given path.
    Because the list_PS_orderfunction only searches the latest 100 orders, this function can only take maximum 100 orders each time

    Args:
        path_list (list): A data path list of MLFluv data that requires labeling 
    """   

    for idx, point_folder in enumerate(path_list):

        point_id = os.path.basename(point_folder)

        print(f"Start to search for {idx} point {point_id}")

        meta_csv_path = [os.path.join(point_folder, file) for file in os.listdir(point_folder) if file.endswith('.csv')][0]

        meta_df = pd.read_csv(meta_csv_path)

        projection = json.loads(meta_df['projection'][0].replace("\'", "\""))
        crs = projection['crs']
        aoi_coords = json.loads(meta_df['aoi'][0])

        # Construct a geometry for Planet Data API
        geometry = {}
        geometry["type"] = "Polygon"
        geometry["coordinates"] = aoi_coords

        try:
            s2_date = datetime.date.fromisoformat(meta_df['s2_date'][0])
        except ValueError as err:
            print(err)
            print("Re-formatting the date string...")
            s2_date = datetime.date.fromisoformat(datetime.datetime.strptime(meta_df['s2_date'][0], '%m/%d/%Y').strftime('%Y-%m-%d'))

        date_start = s2_date
        date_end = s2_date + datetime.timedelta(days=1)

        result_list = search_planet_data(f'{point_id}', geometry, date_start, date_end)

        if len(result_list) == 0:
            print(f"No scene for point {point_id} is detected. Let's expand the date search range to 7 days before to 7 days after the given date.")
            date_start = s2_date - datetime.timedelta(days=7)
            date_end = s2_date + datetime.timedelta(days=7)

            result_list = search_planet_data(f'{point_id}', geometry, date_start, date_end)

            if len(result_list) == 0:
                print(f"No scene for point {point_id} is detected. Let's expand the date search range to 15 days before to 15 days after the given date.")
                date_start = s2_date - datetime.timedelta(days=7)
                date_end = s2_date + datetime.timedelta(days=7)

                result_list = search_planet_data(f'{point_id}', geometry, date_start, date_end)

                if len(result_list) == 0:
                    print("No clear Planet data available within one month, skip this point.")
                    continue # skip the rest of the code i a for loop because we can't find data

        # Get a pd dataframe of all available scenes
        ps_scenes = items_to_scenes(result_list)

        # sort the filtered scenes by cloud 
        ps_scenes.sort_values(by='cloud_cover', ascending=True)    

        scene_ids = ps_scenes['id'].tolist()
        print('Length of filtered scenes are :', len(scene_ids))

        
        # Order scenes that by scene_ids
        orders_url = 'https://api.planet.com/compute/ops/orders/v2'
        # set content type to json
        headers = {'content-type': 'application/json'}

        # only assets from 'analytic_8b_sr_udm2', 'analytic_sr_udm2' bundles can be harmonized
        bundles = ['analytic_8b_sr_udm2', 'analytic_sr_udm2']
        tools = [
            # order_request.toar_tool(scale_factor=10000), # toar_tool only works for the "analytic" bundle
            order_request.reproject_tool(projection='EPSG:4326', kernel='cubic'),
            order_request.composite_tool(), 
            order_request.clip_tool(aoi=geometry),
            order_request.file_format_tool(file_format="COG"),
            order_request.harmonize_tool(target_sensor="Sentinel-2")
        ]

        write_out_dir = os.path.join(point_folder, f'{point_id}_PSScene')

        if not os.path.isdir(write_out_dir):
            # For the case a order has never been made
            os.mkdir(write_out_dir)

        if os.path.isdir(write_out_dir) and len(os.listdir(write_out_dir)) != 0:
            # The case a order has been made and the data has been downloaded
            print(f'Event data for {point_id} already exists, Skip.')
            pass

        elif os.path.isdir(write_out_dir) and len(os.listdir(write_out_dir)) == 0:
            # Be aware this line can return maximum 100 lines of results. 
            # If a list is longer than this, it only counts the latest 100 orders. 
            orders_list = asyncio.run(list_PS_order())
            repeated_order = []
            for order in orders_list:
                # print(order['name'])
                if order['name'] == f"{point_id}_PSScene" and order['state'] == "success":
                    repeated_order.append(order['id'])

            if len(repeated_order) >= 1:
                # Because I have made repeated orders for the same event, only download the first one in the repeated_list
                asyncio.run(download_order(repeated_order[0], dir=write_out_dir))
                print(f'Data for {point_id} Downloaded.')

            else:
                try: 
                    bundle_name = bundles[0]
                    products = [order_request.product(scene_ids, bundle_name, 'PSScene')]
                    request = order_request.build_request(f"{point_id}_PSScene", products=products, tools=tools)    
                    # Creating order, waiting and downloading
                    asyncio.run(create_PS_order(request=request))

                except planet.exceptions.BadQuery: 
                    # this error means the product bundle_name does not exist for this aoi and date 
                    bundle_name = bundles[1]
                    products = [order_request.product(scene_ids, bundle_name, 'PSScene')]
                    request = order_request.build_request(f"{point_id}_PSScene", products=products, tools=tools)
                    
                    try: 
                        asyncio.run(create_PS_order(request=request))
                    
                    except:
                        # We might still get BadQuery, probably because planet does not have data 
                        print('Suspect no harmonization tool available for chosen images. Update filter tool and try download again.')
                        pass                            


if __name__ == "__main__":

    API_KEY = os.environ.get('PL_API_KEY', '')
    print(API_KEY)

    client = Auth.from_key(API_KEY)

    data_path = '/exports/csce/datastore/geos/users/s2135982/MLFLUV_DATA/mlfluv_s12lulc_data_water_from_sediment_rich_sample'
    point_path_list = [os.path.join(data_path, folder) for folder in os.listdir(data_path)]

    sub_lists = [point_path_list[x:x+100] for x in range(0, len(point_path_list), 100)]
    print(len(sub_lists))


    # TODO batch the following process to download 100 points each time

    # Run this multiple times (at least twice) until all data (maximum 100) is downloaded. 
    # Then switch to a new sub list 
    order_100_data(sub_lists[5])
    # print()









