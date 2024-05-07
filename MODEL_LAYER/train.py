# Train for the initial model of Unet 

import csv
import os
import shutil
import json
import numpy as np
from loguru import logger
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn

from model import SMPUnet
from dataset import MLFluvDataset
from interface import MLFluvUnetInterface
from weight_calculator import get_class_weight


def parse_config_params(config_file):
    with open(config_file, 'r') as f:
        config_params = json.load(f)
    return config_params


if __name__ == "__main__":

    ####################################
    # PARSE CONFIG FILE
    ####################################
    print(os.getcwd())
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
    temperature = config_params["model"]['temperature']
    distill_lamda = config_params["model"]['distill_lamda']

    label_mode = config_params['data_loader']['label']

    weights_path = f"MODEL_LAYER/{weight_func}_weights_{label_mode}.csv"

    print(f"Train for log {log_num}")

    # LOGGING

    logger.add(f'experiments/{config_params["trainer"]["log_num"]}/info.log')
    # writer = SummaryWriter(f'./experiments/{log_num}/tensorboard')

    os.makedirs(f'./experiments/{log_num}', exist_ok=True)
    os.makedirs(f'./experiments/{log_num}/checkpoints', exist_ok=True)

    shutil.copy('MODEL_LAYER/config.json', os.path.join(f'./experiments/{log_num}', 'config.json'))
    shutil.copy(f'MODEL_LAYER/dataset.py', os.path.join(f'./experiments/{log_num}', f'dataset.py'))
    shutil.copy(f'MODEL_LAYER/train.py', os.path.join(f'./experiments/{log_num}', f'train.py'))

    # MODEL PARAMS

    ENCODER = config_params['model']['encoder']
    # ENCODER_WEIGHTS = 'imagenet' #None
    # ACTIVATION = None  # could be None for logits (binary) or 'softmax2d' for multicalss segmentation

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = SMPUnet(encoder_name=ENCODER, in_channels=15, num_classes=6)
    # print(model)

    train_set = MLFluvDataset(
        data_path=config_params['data_loader']['args']['train_paths'],
        mode='train',
        folds = [0, 1, 2, 3],
        window=window_size,
        label=label_mode,
        one_hot_encode=False
    )

    val_set = MLFluvDataset(
        data_path=config_params['data_loader']['args']['train_paths'],
        mode='val',
        folds = [4],
        window=window_size, 
        label=label_mode,
        one_hot_encode=False
    )

    # Use saved weights for loss function, if the weights are pre-calculated 
    if os.path.isfile(weights_path):
        df = pd.read_csv(weights_path)
        class_weights = df['Weights']
    else:
        print('Going to calculate weight now..')
        class_weights = get_class_weight(train_set, weight_func=weight_func)
    print(class_weights)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # SET LOSS, OPTIMIZER
    if loss_func == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(reduction='mean',
                                        weight=weights,
                                        label_smoothing=0.01, 
                                        ignore_index=0)
    elif loss_func == "FocalLoss":
        criterion = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='focal_loss',
            alpha=weights,
            gamma=2,
            reduction='mean',
            device=device,
            dtype=torch.float32,
            force_reload=False
        )

    # criterion = smp.losses.DiceLoss(mode='multiclass')
    
    optimiser = optim.Adam(model.parameters(), lr=lr)

    interface = MLFluvUnetInterface(
        model=model,
        data_train=train_set,
        data_val=val_set,
        loss_fn=criterion,
        optimiser=optimiser,
        device=device,
        batch_size=batch_size,
        log_num=log_num,
        distill_lamda = 0,
    )
    
    interface.train(epochs=epochs, eval_interval=10)
