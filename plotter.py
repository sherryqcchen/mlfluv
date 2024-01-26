import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import cv2

# plt.style.use('QC_publication')
# sns.axes_style("darkgrid")

def plot_lulc(data, title='LULC Class Map'):
    """
    Plot a LULC map with merged and edited classes for MLFluv dataset.
    It assumes maps has 8 classes with values varing from 0 - 7 

    Args:
        - data (np.array): 2D array representing the LULC class map where each element 
                            indicates the class value (0-7).
    """

    lulc_cmap = mpl.colors.ListedColormap(['#4183C4', '#009600', '#CCFF99', '#F096FF', '#FA0000', '#FFBB22', '#B4B4B4', '#6BF5FF'])

    plt.imshow(data, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=7)

    # Add color bar for reference
    cbar = plt.colorbar(ticks=np.arange(0, 8))
    class_names = ['water', 'tree', 'shallow-rooted vegetation', 'crops', 'build-up', 'fluvial sediment', 'bare', 'ice/snow']
    cbar.set_label('Class')
    cbar.set_ticklabels(class_names)

    # Set the plot title
    plt.title(title)

    plt.show()

def plot_s2_rgb(s2_array):
    """
    Plot Sentinel-2 RGB bands. 

    Args:
        s2_arr (numoy.ndarray):3D array representing the Sentinel-2 image, where the third dimension
         corresponds to different spectral bands (e.g., bands 4, 3, 2 (index position 3, 2, 1) for RGB)                       
    """    
    rgb_img = s2_array[:, :, [3,2,1]]
    rgb_arr = cv2.normalize(rgb_img,
                            dst=None,
                            alpha=0,
                            beta=255,
                            norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    plt.imshow(rgb_arr)
    plt.show()

def plot_s1(s1_array, vis_option='VV'):
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

    if vis_option=='VV':
        plt.imshow(vv)
    elif vis_option=='VH':
        plt.imshow(vh)
    else:
        # add ratio band for visualization
        ratio = vv / vh

        s1_stack = np.stack([vv, vh, ratio], axis=-1)

        # # Normalize the valuse of three bands
        # s1_stack_reshaped = s1_stack.reshape(-1, 3) 
        # s1_stack_normalized = normalize(s1_stack_reshaped, axis=0).reshape(s1_stack.shape)

        rgb_arr = cv2.normalize(s1_stack,
                                dst=None,
                                alpha=0,
                                beta=255,
                                norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        plt.imshow(rgb_arr)
    plt.show()

def plot_full_data(s1_array, s2_array, esri_array, esawc_array, dw_array, glc10_array, meta_info, savefig=False, fig_name=None):
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
    s2 = cv2.normalize(s2_array[:, :, [3,2,1]],
                        dst=None,
                        alpha=0,
                        beta=255,
                        norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    s1 = s1_array[:,:,0]
    
    lulc_cmap = mpl.colors.ListedColormap(['#4183C4', '#009600', '#CCFF99', '#F096FF', '#FA0000', '#FFBB22', '#B4B4B4', '#6BF5FF'])

    # plt.imshow(data, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=7)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

    for i, ax in enumerate(axes.flat):
        if i == 0:
            im = ax.imshow(s1)
            ax.set_title('Sentinel-1 VV polarization', fontsize=20)
        elif i == 1:
            im = ax.imshow(s2)
            ax.set_title('Sentinel-2 RGB stack', fontsize=20)
        elif i == 2:
            im = ax.imshow(esri_array, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=7)
            ax.set_title('ESRI label', fontsize=20)
        elif i == 3:
            im = ax.imshow(esawc_array, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=7)
            ax.set_title('ESA World Cover label', fontsize=20)
        elif i == 4:
            im = ax.imshow(dw_array, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=7)
            ax.set_title('Dynamic World label', fontsize=20)
        elif i == 5:
            im = ax.imshow(glc10_array, cmap=lulc_cmap, interpolation='none', vmin=0, vmax=7)
            ax.set_title('FROM-GLC10 label', fontsize=20)

    # Add color bar for reference
    # divider = make_axes_locatable(axes[1,1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.subplots_adjust(top=0.9, right=0.8)
    cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(0, 8))
    class_names = ['water', 'tree', 'shallow vege', 'crops', 'build', 'fluvial sediment', 'bare', 'ice/snow']
    cbar.ax.set_title('Class', fontsize=20)
    cbar.set_ticklabels(class_names)
    cbar.ax.tick_params(labelsize=16)

    # TODO Add test to the figure to print meta data of this point
    point_coords = meta_info['point'][0]
    year = meta_info['year'][0]
    river_order = meta_info['riv_order'][0]
    da = meta_info['drainage_area'][0]

    meta_str = f"Images for {point_coords} in {year}.\n River order: {river_order}, upland drainage area: {da} km \u00B2."

    fig.text(0.15, 0.95, meta_str, fontsize=20)

    if savefig==True:
        plt.savefig(f'data_figures/{fig_name}.png')
    
    plt.cla()
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
