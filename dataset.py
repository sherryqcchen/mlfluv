import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import rioxarray
import os
import random
import matplotlib.pyplot as plt

def plot_pair(image, mask, surfix):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image[4,:,:])
    axes[0].set_title('rgb')

    # axes[1].imshow(y_pred.cpu().numpy()[i], cmap='jet')
    # axes[1].set_title('y_val_pred')

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

    # plot_pair(image, mask, 'before_rotate')

    if random.random() > 0.1:
        image = np.rot90(image, axes=(1, 2))
        mask = np.rot90(mask)

        # plot_pair(image, mask, 'after_rotate')

    return image, mask

def flip(image, mask):

    # plot_pair(image, mask, 'before_flip')

    if random.random() > .5:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=1)
    else:
        image = np.flip(image, axis=3)
        mask = np.flip(mask, axis=2)

    # plot_pair(image, mask, 'after_flip')

    return image, mask



def random_crop(image, mask, window=256):
    # plot_pair(image, mask, 'before_crop')

    _, h, w = image.shape

    w_start = random.randint(0, w - window - 1)
    h_start = random.randint(0, h - window - 1)
    image = image[:, h_start:h_start + window, w_start:w_start + window]
    mask = mask[h_start:h_start + window, w_start:w_start + window]

    # plot_pair(image, mask, 'after_crop')

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
            data_path="/exports/csce/datastore/geos/groups/LSDTopoData/MLFluv/mlfluv_5_folds",
            window = 256,
            norm = True,
            mode = 'train',
            folds = [0, 1, 2, 4],
            label = 'hand',
            one_hot_encode = False          
    ):
        """
        Pytorch Dataset class to load samples from the MLFLuv dataset for fluvial system semantic segmentation.

        """   
        # print(os.listdir(data_path))
        self.file_paths = [os.path.join(data_path, file) for file in os.listdir(data_path)] # 5 npy files
        self.all_folds = [np.load(file) for file in self.file_paths] # len() is 5 because of 5 folds split

        if folds == None:
            # If folds are not specified, all data will be loaded
            self.data = np.concatenate([self.all_folds[idx] for idx in range(len(self.all_folds))], axis=0)
        else:
            self.data = np.concatenate([self.all_folds[idx] for idx in folds], axis=0)

        
        self.window = window
        self.mode = mode
        self.norm = norm
        self.label = label
        self.one_hot_encode = one_hot_encode

        if self.one_hot_encode:
            self.label_values = [0, 1, 2, 3, 4, 5, 6, 7]
            

    def transform(self, image, mask, rough_mask=None):

        bands, h, w = image.shape

        if self.mode == 'train':
            if random.random() > .7:
                image, mask = rotate_90_degrees(image, mask)
            if random.random() > .5:
                image, mask = flip(image, mask)

            # random crop 256x256
            if self.window == 512:
                pass
            else:               
                image, mask = random_crop(image, mask, window=self.window)

            image = normalize_per_channel(image)

            # Random mask image stacks
            # image = random_mask(image)

        elif self.mode == 'val':
            if self.window == 512:
                pass
            else:      
                # center crop no rotation (so that val/test are always the same)
                image, mask = center_crop(image, mask, window=self.window)
            image = normalize_per_channel(image)
        else:
            image = normalize_per_channel(image) # do not crop testing set

        image = np.transpose(image, )

        return image, mask

    def __getitem__(self, index):

        data_paths = self.data[index]

        # if the input data is changed, go to split_data.py to check the new orders of s1, s2 and labels

        s1_path = data_paths[2]
        s2_path = data_paths[3]

        auto_mask = data_paths[0]
        hand_mask = data_paths[1]

        s1_arr = np.load(s1_path) # shape [h, w, band], band=2
        s2_arr = np.load(s2_path) # shape [h, w, band], band=13

        auto_mask_arr = np.load(auto_mask).squeeze()[:512, :512]  
        hand_mask_arr = rioxarray.open_rasterio(hand_mask).data.squeeze()[:512, :512]

        if self.label == 'hand':
            mask = hand_mask_arr
        else:
            mask = auto_mask_arr

        # mask no data label as clouds (the class that has not shown in the dataset yet)
        mask = np.where(mask == -999, 7, mask)    

        # Train on S1 2 bands and S2 13 bands
        # clip each image to 512*512 as height * width
        image = np.dstack((s1_arr, s2_arr))[:512, :512, :]  # shape [h, w, band], band=15
        image = np.transpose(image, (2, 0, 1))  # shape [band, h, w], band=15
        
        # Train only on Sen2 13 bands
        # image = s2_image

        image, mask = self.transform(image, mask) # image shape [windows_size, window_size, band], band=15

        # one-hot-encode the mask
        if self.one_hot_encode:
            class_idx = [idx for idx in self.label_values]
            masks = [(mask == idx) for idx in class_idx]
            mask = np.stack(masks, axis = -1) # shape [h, w, band], band=8

            mask = np.transpose(mask, (2, 0, 1)) # shape [band, h, w], band=8

        image = np.transpose(image, (2, 0, 1)) # shape [15, 256, 256]
        # mask = np.transpose(mask, (2, 0, 1)) # shape [band, h, w], band=8

        image = torch.from_numpy(image).float()
        mask = mask.astype("float")
        mask = torch.from_numpy(mask.copy()).long()

        return image, mask
    
    def __len__(self):

        return len(self.data)


if __name__ == '__main__':

    my_dataset = MLFluvDataset(folds=None)
    print(len(my_dataset.data))
    # print(my_dataset.data[0])

    for idx, (image, label) in enumerate(my_dataset):
        # print(idx)
        print(image.shape)
        print(label.dtype)
