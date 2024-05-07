import csv
import os
import numpy as np
import pandas as pd
import json
import sklearn

from dataset import MLFluvDataset

def parse_config_params(config_file):
    with open(config_file, 'r') as f:
        config_params = json.load(f)
    return config_params

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

        # the weight for the background class 0 (clouds, snow/ice and no data) is not needed, so it should be zero out.
        weight[0] = 0

        print(classes)

        # Create a DataFrame from the weights per class
        df = pd.DataFrame({'Class': classes, 'Weights': weight})

        # Create a complete list of class values
        complete_class_values = list(range(min(classes), max(classes) + 1))

        # Create a new DataFrame with complete class values
        complete_df = pd.DataFrame({'Class': complete_class_values})

        # Merge the two DataFrames and fill missing weights with 0
        df = pd.merge(complete_df, df, on='Class', how='left').fillna(0)

        # Save to CSV
        df.to_csv(f'MODEL_LAYER/{weight_func}_weights_{suffix}.csv', index=False)

        return df['Weights']


if __name__ == "__main__":

    ####################################
    # PARSE CONFIG FILE
    ####################################
    config_params = parse_config_params('MODEL_LAYER/config.json')

    log_num = config_params["trainer"]["log_num"]
    in_channels = config_params["trainer"]["in_channels"]
    num_classes = config_params["trainer"]["classes"]
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
    lr = config_params["trainer"]["learning_rate"]
    batch_size = config_params["trainer"]["batch_size"]
    window_size = config_params["trainer"]["window_size"]
    weight_func = config_params["model"]["weights"]
    loss_func = config_params["model"]['loss_function']
    label_mode = config_params['data_loader']['label']
    

    weights_path = f"{weight_func}_weights_{label_mode}.csv"

    train_set = MLFluvDataset(
    data_path=config_params['data_loader']['args']['data_paths'],
    mode='train',
    folds = [0, 1, 2, 3],
    window=window_size,
    label=label_mode,
    one_hot_encode=False
    )

    if os.path.isfile(weights_path):
        class_weights = list(csv.reader(open(weights_path, "r"), delimiter=","))
        class_weights = np.array([float(i) for i in class_weights[0]])
    else:
        class_weights = get_class_weight(train_set, weight_func=weight_func, suffix=label_mode)
    print(class_weights)

