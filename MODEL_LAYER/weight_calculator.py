import argparse
import csv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import pandas as pd
import json
import sklearn

from dataset import MLFluvDataset
from UTILS import utils

root_path = ''
root_path, is_vm = utils.update_root_path_for_machine(root_path=root_path)

if is_vm:
    config_path = os.path.join(root_path,'script/config.yml')
else:
    config_path = os.path.join(root_path, 'script/config_k8s.yml')

def get_class_weight(dataset, weight_func='inverse_log', suffix='auto'):
        # get weights based on the pixel count of each class in train set 
        # calculation refer to a post: 
        # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
        pixel_sum = 0
        class_counts = np.zeros(256)  # Assuming labels are in the range [0, 255]

        for _, label in dataset:
            pixel_sum += np.prod(label.shape)
            unique_classes, counts = np.unique(label, return_counts=True)
            class_counts[unique_classes] += counts

        classes = np.where(class_counts > 0)[0]
        frequencies = np.where(class_counts > 0)[0]
        frequencies = class_counts[classes] 
        num_classes = len(classes)

        class_percent = frequencies / pixel_sum
        print(f"class percent: {class_percent}")

        if weight_func == 'inverse_log':
            weight = pixel_sum / (len(classes) * np.log(frequencies.astype(np.float64)))
        elif weight_func == 'inverse_log_percent':
            weight = 1 / np.log(class_percent.astype(np.float64))
        elif weight_func == 'inverse_count': 
            weight = pixel_sum / (len(classes) * frequencies.astype(np.float64))
        elif weight_func == 'inverse_sqrt':
            weight = pixel_sum / (len(classes) * np.sqrt(class_percent.astype(np.float64)))
        else: 
            print('No weight function is given. We use sklearn compute class weight function')
            weight = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=np.repeat(classes, frequencies))

        print(classes)

        # Create a DataFrame from the weights per class
        df = pd.DataFrame({'Class': classes, 'Weights': weight})

        # Create a complete list of class values
        complete_class_values = list(range(min(classes), max(classes) + 1))

        # Create a new DataFrame with complete class values
        complete_df = pd.DataFrame({'Class': complete_class_values})

        # Merge the two DataFrames and 
        df = pd.merge(complete_df, df, on='Class', how='left')

        # Fill missing weights with the mean weight of all classes exclusing the weight at index 0
        mean_weight = df.loc[df.index != 0, 'Weights'].mean()
        df = df.fillna(mean_weight)   # fillna(0) fill with zero

                
        if 0 in df['Class'].values:
            # If 0 is present, set its weight to 0
            df.loc[df['Class'] == 0, 'Weights'] = 0
        else:
            # If 0 is not present, append a new row with Class = 0 and Weights = 0
            new_row = pd.DataFrame({'Class': [0], 'Weights': [0]})
            df = pd.concat([df, new_row], ignore_index=True)

        # Save to CSV
        df.to_csv(os.path.join(root_path, f'script/MODEL_LAYER/{weight_func}_weights_{suffix}.csv'), index=False)

        return df['Weights']


if __name__ == "__main__":

    ####################################
    # PARSE CONFIG FILE
    ####################################
    parser = argparse.ArgumentParser(description="Please provide a configuration ymal file for trainning a U-Net model.")
    parser.add_argument('--config_path',type=str, default=config_path, help='Path to a configuration yaml file.' )

    args = parser.parse_args()
    config_params = utils.load_config(args.config_path)

    sample_mode = config_params["sample"]["sample_mode"]
    log_num = config_params["trainer"]["log_num"]
    in_channels = config_params["trainer"]["in_channels"]
    num_classes = config_params["trainer"]["classes"]
    train_fold = config_params["trainer"]["train_fold"]
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
    lr = config_params["trainer"]["learning_rate"]
    batch_size = config_params["trainer"]["batch_size"]
    window_size = config_params["trainer"]["window_size"]
    weight_func = config_params["model"]["weights"]
    loss_func = config_params["model"]['loss_function']
    which_label = config_params['data_loader']['which_label']

    # fold_data_path = os.path.join(config_params['data_loader']['train_paths'], f'{sample_mode}_sampling_{which_label}_5_fold')
    # fold_data_path = os.path.join(config_params['data_loader']['train_paths'], f'finetune_with_urban_{which_label}_5_fold')
    fold_data_path = os.path.join(config_params['data_loader']['train_paths'], f'test_{which_label}_fold')

    # weights_path = f"{weight_func}_weights_{which_label}.csv"
    weights_path = "{weight_func}_weights_testset_hand.csv"


    train_set = MLFluvDataset(
        data_path = fold_data_path,
        mode = 'test', # Use val here because we don't want any cropped data for calculating weights
        folds = [0],
        window = window_size,
        label = 'hand',# which_label,
        one_hot_encode = False)

    if os.path.isfile(weights_path):
        class_weights = list(csv.reader(open(weights_path, "r"), delimiter=","))
        class_weights = np.array([float(i) for i in class_weights[0]])
    else:
        class_weights = get_class_weight(train_set, weight_func=weight_func, suffix=f'{which_label}_incre')
    print(class_weights)

