import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np
import cv2



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

if __name__=='__main__':

    data = np.load('20200108T150719_20200108T150715_T18MZU_590S7209W_ESAWC.npy')
    s2_data = np.load('20200108T150719_20200108T150715_T18MZU_590S7209W_S2.npy')
    s1_data = np.load('20200108T150719_20200108T150715_T18MZU_590S7209W_S1.npy')
    plot_lulc(data, 'ESAWC')
    plot_s2_rgb(s2_data)
    plot_s1(s1_data, 'vv')
    plot_s1(s1_data, 'vh')
    plot_s1(s1_data, 'ratio')
