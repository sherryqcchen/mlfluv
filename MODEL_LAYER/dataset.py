import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import rioxarray
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure the script directory is on sys.path so UTILS can be imported reliably
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from UTILS import plotter
from UTILS import interpolation

def plot_pair(image, mask, surfix):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image[4,:,:])
    axes[0].set_title('rgb')

    axes[1].imshow(mask, cmap='jet')
    axes[1].set_title('mask')

    plt.savefig(f'debug_plots/dataset_{surfix}.png')
    plt.close()

def normalize_per_channel(image):

    placeholder = np.zeros_like(image).astype(float)
    for band_id in range(image.shape[0]):
        band = image[band_id, :, :]
        norm_band = (band - band.mean()) / (band.std() + 0.00000001)
        # norm_band = band/255.0

        placeholder[band_id, :, :] = norm_band

    return placeholder



def rotate_90_degrees(image, mask):

    if random.random() > 0.1:
        image = np.rot90(image, axes=(1, 2))
        mask = np.rot90(mask)

    return image, mask

def flip(image, mask):
    # plot_pair(image, mask, "before_flip")

    if random.random() > .5:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=1)
    else:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=0)
    # plot_pair(image, mask, "after_flip")
    return image, mask



def random_crop(image, mask, window=256):

    _, h, w = image.shape

    w_start = random.randint(0, w - window - 1)
    h_start = random.randint(0, h - window - 1)
    image = image[:, h_start:h_start + window, w_start:w_start + window]
    mask = mask[h_start:h_start + window, w_start:w_start + window]

    return image, mask


def random_mask(image, size=20, mask_prob=0.4): #prob 0.8, size 30 was not good in prediction, so change to prob 0.6, size 20
    
    _, h, w = image.shape

    placeholder = np.zeros_like(image).astype(np.float64)
    # for band_id in range(image.shape[-1]):
    for band_id in range(2,15):
        masked_band = image[:, :,band_id] #TAKES ONLY 13 BANDS
        if random.random() <= mask_prob:  # random mask area 50x50
            h_start = random.randint(0, h - size - 1)
            w_start = random.randint(0, w - size - 1)
            # masked_band[h_start:h_start + size, w_start:w_start + size,:] = np.zeros(shape=(size, size,13))
            masked_band[h_start:h_start + size, w_start:w_start + size] = np.zeros(shape=(size, size))
        placeholder[:, :, band_id] = masked_band
    # print(placeholder)
    return placeholder

def center_crop(image, mask, window=192):
   
    y, x = mask.shape
    startx = x // 2 - (window // 2)
    starty = y // 2 - (window // 2)
    image = image[:, starty:starty + window, startx:startx + window]
    mask = mask[starty:starty + window, startx:startx + window]

    return image, mask


class MLFluvDataset(Dataset):

    def __init__(
            self,
            data_path="data/fold_data",
            window_size = 512,
            patch_size = 512,
            norm = True,
            mode = 'train',
            folds = [0, 1, 2, 3],
            label = None,
            one_hot_encode = False,
            bands = ['VV','VH','B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'],    # 'B10',
            debug_nan=False,
            nan_debug_dir='debug_plots/nan_masks',
            nan_debug_limit=5,
            nan_overlay_source='s2',
            nan_overlay_pol='VV',
            nan_handling='mask',
    ):
        """
        Pytorch Dataset class to load samples from the MLFLuv dataset for fluvial system semantic segmentation.

        """   
        
        # print(os.listdir(data_path))
        self.file_paths = [os.path.join(data_path, file) for file in os.listdir(data_path)] # 5 npy files
        self.all_folds = [np.load(file, allow_pickle=True) for file in self.file_paths if file.endswith('.npy')] # len() is 5 because of 5 folds split

        if folds == None:
            # If folds are not specified, all data will be loaded
            self.data = np.concatenate([self.all_folds[idx] for idx in range(len(self.all_folds))], axis=0)
        else:
            self.data = np.concatenate([self.all_folds[idx] for idx in folds], axis=0)

        self.s1_bands = ['VV', 'VH']
        self.s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'] # 'B10'
        self.all_bands = self.s1_bands + self.s2_bands  # Full list of 15 bands
        self.bands = bands
        self.window_size = window_size
        self.patch_size = patch_size
        self.mode = mode
        self.norm = norm
        self.label = label
        self.one_hot_encode = one_hot_encode
        self.debug_nan = debug_nan
        self.nan_debug_dir = nan_debug_dir
        self.nan_debug_limit = nan_debug_limit
        self.nan_overlay_source = nan_overlay_source.lower()
        self.nan_overlay_pol = nan_overlay_pol
        if self.nan_overlay_source not in {'s1', 's2'}:
            self.nan_overlay_source = 's2'
        valid_nan_policies = {'mask', 'drop', 'interpolate'}
        if nan_handling not in valid_nan_policies:
            raise ValueError(f"nan_handling must be one of {valid_nan_policies}, got '{nan_handling}'")
        self._nan_debug_count = 0
        self._nan_skip_count = 0
        self.nan_handling = nan_handling

        if self.one_hot_encode:
            self.label_values = [0, 1, 2, 3, 4, 5, 6]
    
    def get_band_indices(self):
        """
        Get the indices of selected bands from the full list of available bands.
        """
        band_indices = {band: i for i, band in enumerate(self.all_bands)}
        return [band_indices[band] for band in self.bands if band in band_indices]
            

    def transform(self, image, mask, rough_mask=None):

        bands, h, w = image.shape

        if self.mode == 'train':
            if random.random() > .7:
                image, mask = rotate_90_degrees(image, mask)
            if random.random() > .5:
                image, mask = flip(image, mask)

            # random crop 256x256
            if self.window_size == self.patch_size:
                pass
            else:               
                image, mask = random_crop(image, mask, window=self.window_size) 

            image = normalize_per_channel(image)

            # Random mask image stacks
            # image = random_mask(image)

        elif self.mode == 'val':
            if self.window_size == self.patch_size:
                pass
            else:      
                # center crop no rotation (so that val/test are always the same)
                image, mask = center_crop(image, mask, window=self.window_size)
            image = normalize_per_channel(image)
        else:   
            image = normalize_per_channel(image) # do not crop testing set

        return image, mask

    def _visualize_nan_patch(self, patch_s1, patch_s2, union_mask, index, metadata=None):
        """
        Save debug plots showing where NaNs exist within the current patch.
        """
        if not self.debug_nan:
            return
        if self._nan_debug_count >= self.nan_debug_limit:
            return

        os.makedirs(self.nan_debug_dir, exist_ok=True)
        base_name = os.path.join(self.nan_debug_dir, f'idx_{index:05d}')

        overlay_path = f'{base_name}_nan_overlay.png'
        if self.nan_overlay_source == 's1':
            plotter.plot_nan_overlay_s1(
                patch_s1,
                union_mask,
                save_path=overlay_path,
                title=f'NaN overlay on Sentinel-1 {self.nan_overlay_pol}',
                polarization=self.nan_overlay_pol
            )
        else:
            plotter.plot_nan_overlay(
                patch_s2,
                union_mask,
                save_path=overlay_path,
                title='NaN overlay on Sentinel-2 RGB'
            )

        if metadata:
            info_path = f'{base_name}_paths.txt'
            with open(info_path, 'w') as info_file:
                for key, value in metadata.items():
                    info_file.write(f'{key}: {value}\n')

        self._nan_debug_count += 1

    def _interpolate_patch(self, patch):
        filled_patch = patch.copy()
        for band in range(filled_patch.shape[2]):
            band_data = filled_patch[:, :, band]
            if np.all(np.isfinite(band_data)):
                continue
            interpolated = interpolation.interpolate_nd(band_data)
            filled_patch[:, :, band] = interpolated
        return filled_patch

    def _load_sample(self, index):

        data_paths = self.data[index]

        # if the input data is changed, go to split_data.py to check the new orders of s1, s2 and labels
        s1_path = [path for path in data_paths if path.endswith('S1.npy')][0]
        s2_path = [path for path in data_paths if path.endswith('S2.npy')][0]

        s1_arr = np.load(s1_path) # shape [h, w, band], band=2
        s2_arr = np.load(s2_path) # shape [h, w, band], band=12

        if self.label == 'hand':
            hand_mask = [path for path in data_paths if path.endswith('hand.tif')][0]
            hand_mask_arr = rioxarray.open_rasterio(hand_mask).data.squeeze()[:self.patch_size, :self.patch_size]
            mask = hand_mask_arr
            label_path_used = hand_mask
        else:
            auto_mask = [path for path in data_paths if path.endswith(f'{self.label}.npy')][0]
            auto_mask_arr = np.load(auto_mask).squeeze()[:self.patch_size, :self.patch_size]
            mask = auto_mask_arr
            label_path_used = auto_mask

            if self.mode == 'initial_train':
                mask = np.where(mask == 6, 5, mask)

        # Handle possible invalid data in Sentinel images, mask them in the labels
        s2_arr[(s2_arr<0) | (s2_arr>10000)] = np.nan
        s1_arr[~np.isfinite(s1_arr)] = np.nan

        patch_s1 = s1_arr[:self.patch_size, :self.patch_size, :]
        patch_s2 = s2_arr[:self.patch_size, :self.patch_size, :]

        if np.isnan(patch_s2).any() or np.isnan(patch_s1).any():
            mask_s1_nan = np.isnan(patch_s1).any(axis=2)
            mask_s2_nan = np.isnan(patch_s2).any(axis=2)
            union_mask = np.logical_or(mask_s1_nan, mask_s2_nan)

            if self.debug_nan:
                self._visualize_nan_patch(
                    patch_s1.copy(),
                    patch_s2.copy(),
                    union_mask,
                    index,
                    metadata={
                        's1_path': s1_path,
                        's2_path': s2_path,
                        'label_path': label_path_used
                    }
                )
            if self.nan_handling == 'drop':
                self._nan_skip_count += 1
                if self.debug_nan:
                    print(f"[MLFluvDataset] Skipping sample index {index} due to NaNs (total skipped: {self._nan_skip_count}).")
                return None
            elif self.nan_handling == 'interpolate':
                patch_s1 = self._interpolate_patch(patch_s1)
                patch_s2 = self._interpolate_patch(patch_s2)
                if np.isnan(patch_s1).any() or np.isnan(patch_s2).any():
                    # Fall back to masking if interpolation failed
                    mask[union_mask] = 0
                    patch_s1[union_mask] = 0
                    patch_s2[union_mask] = 0
            else:
                mask[union_mask] = 0
                patch_s1[union_mask] = 0
                patch_s2[union_mask] = 0

        mask = np.where((mask >= 0) & (mask <= 6), mask, 0)
        self.num_classes = 6

        if self.label == 'hand':
            self.num_classes = 7

        full_image = np.dstack((patch_s1, patch_s2))[:self.patch_size, :self.patch_size, :]  # shape [h, w, band], band=15

        if isinstance(full_image, np.ndarray) and np.isnan(full_image).any():
            raise ValueError(f"NaN found in input tensor image before returning for index {index}")

        selected_indices = self.get_band_indices()
        image = full_image[:, :, selected_indices]
        image = np.transpose(image, (2, 0, 1))  # shape [band, h, w], band=15

        image, mask = self.transform(image, mask) # image shape [windows_size, window_size, band], band=15

        if self.one_hot_encode:
            class_idx = [idx for idx in self.label_values]
            masks = [(mask == idx) for idx in class_idx]
            mask = np.stack(masks, axis = -1) # shape [h, w, band], band=8

        image = torch.from_numpy(image).float()
        mask = mask.astype("float")
        mask = torch.from_numpy(mask.copy()).long()

        return image, mask

    def __getitem__(self, index):

        data_len = len(self.data)
        attempts = 0

        while attempts < data_len:
            actual_index = (index + attempts) % data_len
            sample = self._load_sample(actual_index)
            if sample is not None:
                return sample
            attempts += 1

        raise RuntimeError("Unable to fetch a valid sample without NaNs after checking the entire dataset.")
    
    def __len__(self):

        return len(self.data)


if __name__ == '__main__':

    # my_dataset = MLFluvDataset(data_path='', 
    #                            folds=[0], 
    #                            mode='train',
    #                            label='DW',
    #                            bands = ['B2', 'B3', 'B4', 'B8'])
    # print(len(my_dataset.data[0]))
    # # print(my_dataset.data[0])

    # for idx, (image, label) in enumerate(my_dataset):
    #     print(idx)
    #     print(image.shape)
    #     print(np.unique(label))

    dataset = MLFluvDataset(
        data_path='data/fold_data/finetune_DW_5_fold',
        folds=[0, 1, 2, 3, 4],
        label='DW',
        mode='train',
        debug_nan=True,
        nan_debug_limit=5
    )

    for idx in range(min(50, len(dataset))):
        _ = dataset[idx]
