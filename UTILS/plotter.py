import os
from pathlib import Path
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

DEFAULT_DEBUG_DIR = Path('debug_plots/figures')


def _normalize_to_uint8(arr):
    arr = np.asarray(arr, dtype=np.float32)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr, dtype=np.uint8)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if np.isclose(max_val, min_val):
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)
    return (norm * 255).astype(np.uint8)


def _cv2_normalize(arr):
    if cv2 is not None:
        return cv2.normalize(
            arr,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)
    return _normalize_to_uint8(arr)


def _finalize_plot(fig, save_path=None, prefix='fig'):
    """
    Save or display the current matplotlib figure depending on save_path.
    """
    if not save_path:
        DEFAULT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        save_path = DEFAULT_DEBUG_DIR / f'{prefix}_{timestamp}.png'
    else:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_lulc(data, title='LULC Class Map', save_path=None):
    """
    Plot a LULC map with merged and edited classes for MLFluv dataset.
    It assumes maps has 8 classes with values varing from 0 - 7 

    Args:
        - data (np.array): 2D array representing the LULC class map where each element 
                            indicates the class value (0-7).
    """

    lulc_cmap = mpl.colors.ListedColormap(['#6BF5FF', '#009600', '#CCFF99', '#4183C4', '#FA0000','#B4B4B4', '#FFBB22'])

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=6)

    # Add color bar for reference
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, 8))
    class_names = ['background', 'tree', 'shallow-rooted vegetation','water', 'build-up', 'bare', 'fluvial sediment']
    cbar.set_label('Class')
    cbar.set_ticklabels(class_names)

    # Set the plot title
    ax.set_title(title)
    ax.axis('off')

    _finalize_plot(fig, save_path, prefix='lulc')

def plot_s2_rgb(s2_array, save_path=None, title='Sentinel-2 RGB stack'):
    """
    Plot Sentinel-2 RGB bands.

    Args:
        s2_arr (numoy.ndarray):3D array representing the Sentinel-2 image, where the third dimension
         corresponds to different spectral bands (e.g., bands 4, 3, 2 (index position 3, 2, 1) for RGB)                       
    """    
    rgb_img = s2_array[:, :, [3, 2, 1]]
    rgb_arr = _cv2_normalize(rgb_img)
    fig, ax = plt.subplots()
    ax.imshow(rgb_arr)
    ax.set_title(title)
    ax.axis('off')
    _finalize_plot(fig, save_path, prefix='s2_rgb')

def plot_s1(s1_array, vis_option='VV', save_path=None, title=None):
    """
    Plot Sentinel-1 polarizations.

    Args:
        s1_array (numpy.ndarray): 3D array representing the Sentinel-1 radar bands, 
                                where the third dimension corresponds to VV (vertical-vertical) and 
                                VH (vertical-horizontal) bands.
        vis_option (str, optional): Visualization option. 
                                'VV' for plotting only the VV band, 'VH' for plotting only the VH band,
                                or any other value to plot an RGB image with VV, VH, and VV/VH ratio bands.
    """    
    vv = s1_array[:, :, 0]
    vh = s1_array[:, :, 1]

    fig, ax = plt.subplots()
    if vis_option == 'VV':
        ax.imshow(vv)
    elif vis_option == 'VH':
        ax.imshow(vh)
    else:
        ratio = vv / vh
        s1_stack = np.stack([vv, vh, ratio], axis=-1)
        rgb_arr = _cv2_normalize(s1_stack)
        ax.imshow(rgb_arr)
    default_title = f'Sentinel-1 {vis_option.upper()} polarization' if vis_option in ['VV', 'VH'] else 'Sentinel-1 composite'
    ax.set_title(title or default_title)
    ax.axis('off')
    _finalize_plot(fig, save_path, prefix=f's1_{vis_option}')


def plot_nan_mask(nan_mask, save_path=None, title='NaN mask'):
    """
    Visualize a binary mask showing where NaNs were detected.
    """
    fig, ax = plt.subplots()
    cmap = ListedColormap(['black', '#ff8800'])
    ax.imshow(nan_mask.astype(int), cmap=cmap, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    _finalize_plot(fig, save_path, prefix='nan_mask')


def plot_nan_overlay(s2_array, nan_mask, save_path=None, title='NaNs over Sentinel-2 RGB'):
    """
    Overlay the NaN mask on top of an RGB Sentinel-2 visualization.
    """
    rgb_img = _cv2_normalize(s2_array[:, :, [3, 2, 1]])
    fig, ax = plt.subplots()
    ax.imshow(rgb_img)
    masked = np.ma.masked_where(~nan_mask, nan_mask)
    ax.imshow(masked, cmap='autumn', alpha=0.5, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    _finalize_plot(fig, save_path, prefix='nan_overlay')


def plot_nan_overlay_s1(s1_array, nan_mask, save_path=None, title='NaNs over Sentinel-1 VV', polarization='VV'):
    """
    Overlay the NaN mask on top of a Sentinel-1 polarization image.
    """
    pol_index = 0 if polarization.upper() == 'VV' else 1
    pol_img = s1_array[:, :, pol_index]
    fig, ax = plt.subplots()
    ax.imshow(pol_img, cmap='gray')
    masked = np.ma.masked_where(~nan_mask, nan_mask)
    ax.imshow(masked, cmap='autumn', alpha=0.5, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    _finalize_plot(fig, save_path, prefix='nan_overlay_s1')

def plot_s12label(s1_array, s2_array, label_array, meta_info, savefig=False, fig_name=None, which_label='ESRI', s2_vis_false=True):

    s2_rgb = _cv2_normalize(s2_array[:, :, [3,2,1]])
    s2_false_color = _cv2_normalize(s2_array[:, :, [7, 3, 2]])
    s1 = s1_array[:,:,0]
    
    if s2_vis_false:
        s2 = s2_false_color
    else:
        s2 = s2_rgb

    lulc_cmap = mpl.colors.ListedColormap(['#6BF5FF', '#009600', '#CCFF99', '#4183C4', '#FA0000','#B4B4B4', '#FFBB22'])

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for i, ax in enumerate(axes.flat):
        if i == 0:
            im = ax.imshow(s1, cmap='gray')
            ax.set_title('Sentinel-1 VV polarization', fontsize=20)
        elif i == 1:
            im = ax.imshow(s2)
            ax.set_title('Sentinel-2 RGB stack', fontsize=20)
        elif i == 2:
            im = ax.imshow(label_array, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=6)
            ax.set_title(f'Remapped {which_label} label', fontsize=20)
    
     # Add color bar for reference
    fig.subplots_adjust(top=0.8, right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(0, 7))
    class_names = ['background', 'tree', 'shallow-rooted vegetation','water', 'build-up', 'bare', 'fluvial sediment']
    cbar.ax.set_title('Class', fontsize=20)
    cbar.set_ticklabels(class_names)
    cbar.ax.tick_params(labelsize=16)

    # Add test to the figure to print meta data of this point
    point_coords = meta_info['point'][0]
    year = meta_info['year'][0]
    # river_order = meta_info['riv_order'][0]
    # da = meta_info['drainage_area'][0]

    meta_str = f"Images for {point_coords} in {year}."
    #f"Images for {point_coords} in {year}.\n River order: {river_order}, upland drainage area: {da} km \u00B2."

    fig.text(0.10, 0.85, meta_str, fontsize=18)

    if savefig==True:
        plt.savefig(f'data_figure/{fig_name}_fluv.png')
    
    plt.cla()
    plt.close(fig)

    

def plot_full_data(s1_array, s2_array, esri_array, esawc_array, dw_array, glc10_array, meta_info, savefig=False, fig_name=None, s2_vis_false=True):
    """
    Plot Sentinel-1&2 images and 4 types of land cover products, expot to png.

    Args:
        s1_array (numpy.ndarray): sentinel-1 data array
        s2_array (numpy.ndarray): sentinel-2 data array
        esri_array (numpy.2darray):ESRI land cover data array
        esawc_array (numpy.2darray): ESA World Cover land cover data array
        dw_array (numpy.2darray): Dynamic World land cover data array
        glc10_array (numpy.2darray): FROM-GLC10 land cover data array.
        meta_info (pandas.dataframe object): a dataframe stores meta info point coordinates, river order, 
        savefig (bool, optional): Whether save the plot. Defaults to False.
        fig_name (str, optional): svaed figure name. Defaults to None.
    """    
    s2_rgb = _cv2_normalize(s2_array[:, :, [3,2,1]])
    s2_false_color = _cv2_normalize(s2_array[:, :, [7, 3, 2]])
    if s2_vis_false:
        s2 = s2_false_color
    else:
        s2 = s2_rgb

    s1 = s1_array[:,:,0]
    
    lulc_cmap = mpl.colors.ListedColormap(['#6BF5FF', '#009600', '#CCFF99', '#4183C4', '#FA0000','#B4B4B4', '#FFBB22'])

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

    for i, ax in enumerate(axes.flat):
        if i == 0:
            im = ax.imshow(s1, cmap='gray')
            ax.set_title('Sentinel-1 VV polarization', fontsize=20)
        elif i == 1:
            im = ax.imshow(s2)
            ax.set_title('Sentinel-2 false color stack', fontsize=20)
        elif i == 2:
            im = ax.imshow(esri_array, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=6)
            ax.set_title('ESRI label', fontsize=20)
        elif i == 3:
            im = ax.imshow(esawc_array, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=6)
            ax.set_title('ESA World Cover label', fontsize=20)
        elif i == 4:
            im = ax.imshow(dw_array, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=6)
            ax.set_title('Dynamic World label', fontsize=20)
        elif i == 5:
            im = ax.imshow(glc10_array, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=6)
            ax.set_title('FROM-GLC10 label', fontsize=20)

    # Add color bar for reference
    fig.subplots_adjust(top=0.9, right=0.8)
    cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(0, 7))
    class_names = ['background', 'tree', 'shallow-rooted vegetation','water', 'build-up', 'bare', 'fluvial sediment']
    cbar.ax.set_title('Class', fontsize=20)
    cbar.set_ticklabels(class_names)
    cbar.ax.tick_params(labelsize=16)

    # Add test to the figure to print meta data of this point
    point_coords = meta_info['point'][0]
    year = meta_info['year'][0]
    river_order = meta_info['riv_order'][0]
    da = meta_info['drainage_area'][0]
    date = meta_info['s2_date'][0]

    meta_str = f"Images at {date} for {point_coords} in {year}.\n River order: {river_order}, upland drainage area: {da} km \u00B2."

    fig.text(0.15, 0.95, meta_str, fontsize=20)
    
    data_figure_path = 'data_figure'
    os.makedirs(data_figure_path, exist_ok=True)

    if savefig==True:
        plt.savefig(f'{data_figure_path}/{fig_name}.png')
    
    plt.cla()
    plt.close(fig)


def plot_inference_result(sentinel2_rgb_image, sentinel1_vv_image, label_image, prediction_image, out_dir, out_suffix):

    # Set the font size globally
    mpl.rcParams.update({'font.size': 20})

    # Define custom colors for each class
    colors = ['black', 'darkgreen', 'lightyellow', 'blue','red', 'gray','orange']

    # Create a custom color map using ListedColormap
    custom_cmap = ListedColormap(colors)

    # Define a normalization from values --> colors
    norm = mpl.colors.BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6], 7)

    # Create a figure and axes with a shape of [2, 2]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 15))

    # Plot Sentinel-2 RGB image
    axes[0, 0].imshow(sentinel2_rgb_image)
    axes[0, 0].set_title('Sentinel-2 RGB Image')
    axes[0, 0].axis('off')

    # Plot Sentinel-1 VV image
    axes[0, 1].imshow(sentinel1_vv_image, cmap='gray')
    axes[0, 1].set_title('Sentinel-1 VV Image')
    axes[0, 1].axis('off')

    # Plot ground truth label image
    axes[1, 0].imshow(label_image, cmap=custom_cmap, norm=norm)
    axes[1, 0].set_title('Ground Truth Label')
    axes[1, 0].axis('off')

    # Plot predicted segmentation image
    axes[1, 1].imshow(prediction_image, cmap=custom_cmap, norm=norm)
    axes[1, 1].set_title('Predicted Segmentation')
    axes[1, 1].axis('off')

    # Add color legend

    class_labels = ['background', 'tree', 'shallow-rooted vegetation','water', 'build-up', 'bare', 'fluvial sediment']
    legend_patches = [mpatches.Patch(facecolor=color, label=label, edgecolor='black') for color, label in zip(colors, class_labels)]
    plt.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(1, 0.5), fontsize=20)


    # Adjust layout
    # plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.3, hspace=0.4)

    # Save the figure to a file
    plt.savefig(f'{out_dir}/pred_{out_suffix}.png')

    # Close the figure to free up memory
    plt.close(fig)



if __name__=='__main__':

    data = np.load('20200108T150719_20200108T150715_T18MZU_590S7209W_ESAWC.npy')
    s2_data = np.load('20200108T150719_20200108T150715_T18MZU_590S7209W_S2.npy')
    s1_data = np.load('20200108T150719_20200108T150715_T18MZU_590S7209W_S1.npy')
    plot_lulc(data, 'ESAWC')
    plot_s2_rgb(s2_data)
    plot_s1(s1_data, 'VV')
    plot_s1(s1_data, 'VH')
    plot_s1(s1_data, 'ratio')
