import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
import os
from zipfile import ZipFile
from tqdm import tqdm


def parse_config_params(config_file):
    with open(config_file, 'r') as f:
        config_params = json.load(f)
    return config_params


def get_band_stats(image_arr):
    stats = {
        '0': {'mean': 0.0, 'std': 0.0},
        '1': {'mean': 0.0, 'std': 0.0},
        '2': {'mean': 0.0, 'std': 0.0},
        '3': {'mean': 0.0, 'std': 0.0},
        '4': {'mean': 0.0, 'std': 0.0},
        '5': {'mean': 0.0, 'std': 0.0},
        '6': {'mean': 0.0, 'std': 0.0},
        '7': {'mean': 0.0, 'std': 0.0},
        '8': {'mean': 0.0, 'std': 0.0},
        '9': {'mean': 0.0, 'std': 0.0},
        '10': {'mean': 0.0, 'std': 0.0},
        '11': {'mean': 0.0, 'std': 0.0},
        '12': {'mean': 0.0, 'std': 0.0}
    }

    for image_path in image_arr:
        image = np.load(image_path)
        for band_id in range(image.shape[-1]):
            if len(image.shape) == 3:
                band = image[:, :, band_id]
            else:
                band = image[0, :, :, band_id]
            band_mean, band_std = band.mean(), band.std()
            stats[f'{band_id}']['mean'] += band_mean
            stats[f'{band_id}']['std'] += band_std

    for k, v in stats.items():
        try:
            v['mean'] = (v['mean'] / len(image_arr)).astype(np.float32)
            v['std'] = (v['std'] / len(image_arr)).astype(np.float32)
        except Exception as e:
            print(e)
    print()

    return stats


def zip_n_clean(output_folder):
    # zip output files
    output_files = os.listdir(output_folder)
    with ZipFile(os.path.join(output_folder, 'output.zip'), 'w') as zip:
        for output_file in output_files:
            zip.write(os.path.join(output_folder, output_file), arcname=output_file)

    for output_file in output_files:
        os.remove(os.path.join(output_folder, output_file))


def extract_patches(img, patch_size, stride):
    if len(img.shape) == 2:
        h, w = img.shape
        if h < patch_size[0]:
            img = np.pad(img, ((0, abs(h - patch_size[0])), (0, 0)), constant_values=0)
        if w < patch_size[1]:
            img = np.pad(img, ((0, 0), (0, abs(w - patch_size[1]))), constant_values=0)
    else:
        h, w, _ = img.shape
        if h < patch_size[0]:
            img = np.pad(img, ((0, abs(h - patch_size[0])), (0, 0), (0, 0)), constant_values=0)
        if w < patch_size[1]:
            img = np.pad(img, ((0, 0), (0, abs(w - patch_size[1])), (0, 0)), constant_values=0)

    count_occ_map = np.zeros_like(np.transpose(img, (2, 0, 1)))

    padded_img_shape = img.shape
    num_whole_patches_w, residual_w = divmod(w - patch_size[1] + stride[1], stride[1])
    num_whole_patches_h, residual_h = divmod(h - patch_size[0] + stride[0], stride[0])

    w_starts = []
    for w_start in range(num_whole_patches_w):
        tmp = w_start * stride[1]
        w_starts.append(tmp)
    if residual_w != 0:
        w_starts.append(w - patch_size[1])

    h_starts = []
    for h_start in range(num_whole_patches_h):
        tmp = h_start * stride[0]
        h_starts.append(tmp)
    if residual_h != 0:
        h_starts.append(h - patch_size[0])

    patches = []
    hw_combs = []
    for h_start in tqdm(h_starts):
        for w_start in w_starts:
            hw_combs.append((h_start, w_start))
            patches.append(img[h_start:h_start + patch_size[0], w_start:w_start + patch_size[1]])
            count_occ_map[h_start:h_start + patch_size[0], w_start:w_start + patch_size[1]] += 1

    return patches, count_occ_map, hw_combs, padded_img_shape


def reconstruct_from_patches(orig_img_shape, patches, count_occ_map, hw_combs, padded_img_shape, classes=3):
    patch_size = patches[0].shape
    if classes!=1:
        rec_img = np.zeros(shape=(classes, padded_img_shape[0], padded_img_shape[1]), dtype=np.float)
    else:
        rec_img = np.zeros(shape=(padded_img_shape[0], padded_img_shape[1]), dtype=np.float)
    for patch, hw_start in zip(patches, hw_combs):
        h_s, w_s = hw_start
        if len(patch_size) == 2:
            rec_img[h_s:h_s + patch_size[0], w_s:w_s + patch_size[1]] += patch
        elif len(patch_size) == 3:
            rec_img[:, h_s:h_s + patch_size[1], w_s:w_s + patch_size[2]] += patch
    if len(patch_size) == 2:
        rec_img = rec_img / count_occ_map[:, :, 0]
    elif len(patch_size) == 3:
        rec_img = rec_img / count_occ_map[0:classes, :, :]
    if orig_img_shape != padded_img_shape:
        print('')
        rec_img = rec_img[0:orig_img_shape[0], 0:orig_img_shape[1]]

    return rec_img


def reconstruct_from_patches_with_clip(orig_img_shape, patches, hw_combs, padded_img_shape):
    patch_size = patches[0].shape
    rec_img = np.zeros(shape=padded_img_shape, dtype=np.float)
    occ_map = np.zeros(shape=padded_img_shape, dtype=np.float)

    for patch, hw_start in zip(patches, hw_combs):
        h_s, w_s = hw_start
        if h_s != 0:
            h_start = h_s + 100
            ph_start = 100
        else:
            h_start = h_s
            ph_start = 0
        if h_s < padded_img_shape[0] - patch_size[0]:
            h_end = h_s + patch_size[0] - 100
            ph_end = patch_size[0] - 100
        else:
            h_end = h_s + patch_size[0]
            ph_end = patch_size[0]
        if w_s != 0:
            w_start = w_s + 100
            pw_start = 100
        else:
            w_start = w_s
            pw_start = 0
        if w_s < padded_img_shape[1] - patch_size[1]:
            w_end = w_s + patch_size[1] - 100
            pw_end = patch_size[1] - 100
        else:
            w_end = w_s + patch_size[1]
            pw_end = patch_size[1]
        # print(h_start, h_end, w_start, w_end)

        rec_img[h_start:h_end, w_start:w_end] += patch[ph_start:ph_end, pw_start:pw_end]
        occ_map[h_start:h_end, w_start:w_end] += 1

    rec_img = rec_img / occ_map
    if orig_img_shape != padded_img_shape:
        # print('')
        rec_img = rec_img[0:orig_img_shape[0], 0:orig_img_shape[1]]

    return rec_img


def get_video_gsd(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    elem = root.findall('.//GSD/MEDIAN')[0]  # works only for VividX
    gsd = float(elem.text)

    return gsd


def read_video(filepath, first_frame_only=False):
    cap = cv2.VideoCapture(filepath)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if first_frame_only:
        _, frame = cap.read(0)
        return frame, _
    else:
        inp_vol = np.zeros(shape=(num_frames, height, width))

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break

            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert current frame to grayscale
            inp_vol[frame_index - 1, :, :] = gray_frame
        return inp_vol, frame_rate


import rasterio
import xarray as xr
import numpy as np
import pandas as pd
import os


def rasterio_to_xarray(fname):
    """Converts the given file to an xarray.DataArray object.
    Arguments:
     - `fname`: the filename of the rasterio-compatible file to read
    Returns:
        An xarray.DataArray object containing the data from the given file,
        along with the relevant geographic metadata.
    Notes:
    This produces an xarray.DataArray object with two dimensions: x and y.
    The co-ordinates for these dimensions are set based on the geographic
    reference defined in the original file.
    This has only been tested with GeoTIFF files: use other types of files
    at your own risk!"""
    with rasterio.drivers():
        with rasterio.open(fname) as src:
            data = src.read(1)

            # Set values to nan wherever they are equal to the nodata
            # value defined in the input file
            data = np.where(data == src.nodata, np.nan, data)

            # Get coords
            nx, ny = src.width, src.height
            x0, y0 = src.bounds.left, src.bounds.top
            dx, dy = src.res[0], -src.res[1]

            coords = {'y': np.arange(start=y0, stop=(y0 + ny * dy), step=dy),
                      'x': np.arange(start=x0, stop=(x0 + nx * dx), step=dx)}

            dims = ('y', 'x')

            attrs = {}

            try:
                aff = src.affine
                attrs['affine'] = aff.to_gdal()
            except AttributeError:
                pass

            try:
                c = src.crs
                attrs['crs'] = c.to_string()
            except AttributeError:
                pass

    return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)


def xarray_to_rasterio(xa, output_filename):
    """Converts the given xarray.DataArray object to a raster output file
    using rasterio.
    Arguments:
     - `xa`: The xarray.DataArray to convert
     - `output_filename`: the filename to store the output GeoTIFF file in
    Notes:
    Converts the given xarray.DataArray to a GeoTIFF output file using rasterio.
    This function only supports 2D or 3D DataArrays, and GeoTIFF output.
    The input DataArray must have attributes (stored as xa.attrs) specifying
    geographic metadata, or the output will have _no_ geographic information.
    If the DataArray uses dask as the storage backend then this function will
    force a load of the raw data.
    """
    # Forcibly compute the data, to ensure that all of the metadata is
    # the same as the actual data (ie. dtypes are the same etc)
    xa = xa.load()

    if len(xa.shape) == 2:
        count = 1
        height = xa.shape[0]
        width = xa.shape[1]
        band_indicies = 1
    else:
        count = xa.shape[0]
        height = xa.shape[1]
        width = xa.shape[2]
        band_indicies = np.arange(count) + 1

    processed_attrs = {}

    try:
        val = xa.attrs['affine']
        processed_attrs['affine'] = rasterio.Affine.from_gdal(*val)
    except KeyError:
        pass

    try:
        val = xa.attrs['crs']
        processed_attrs['crs'] = rasterio.crs.CRS.from_string(val)
    except KeyError:
        pass

    with rasterio.open(output_filename, 'w',
                       driver='GTiff',
                       height=height, width=width,
                       dtype=str(xa.dtype), count=count,
                       **processed_attrs) as dst:
        dst.write(xa.values, band_indicies)


def xarray_to_rasterio_by_band(xa, output_basename, dim='time', date_format='%Y-%m-%d'):
    for i in range(len(xa[dim])):
        args = {dim: i}
        data = xa.isel(**args)
        index_value = data[dim].values

        if type(index_value) is np.datetime64:
            formatted_index = pd.to_datetime(index_value).strftime(date_format)
        else:
            formatted_index = str(index_value)

        filename = output_basename + formatted_index + '.tif'
        xarray_to_rasterio(data, filename)
        print('Exported %s' % formatted_index)


if __name__ == "__main__":
    pass