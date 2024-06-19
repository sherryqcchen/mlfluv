# Train for the initial model of Unet 

import argparse
import csv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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
from UTILS import utils
from weight_calculator import get_class_weight


if __name__ == "__main__":

    root_path = ''
    root_path, is_vm = utils.update_root_path_for_machine(root_path=root_path)

    if is_vm:
        config_path = os.path.join(root_path,'script/config.yml')
    else:
        config_path = os.path.join(root_path, 'script/config_k8s.yml')

    ####################################
    # PARSE CONFIG FILE
    ####################################
    parser = argparse.ArgumentParser(description="Please provide a configuration ymal file for trainning a U-Net model.")
    parser.add_argument('--config_path',type=str, default=config_path, help='Path to a configuration yaml file.' )

    args = parser.parse_args()
    config_params = utils.load_config(args.config_path)

    sample_mode = config_params["sample"]["sample_mode"]
    which_label = config_params["data_loader"]["which_label"]

    log_num = config_params["trainer"]["log_num"]
    train_fold = config_params["trainer"]["train_fold"]
    valid_fold = config_params["trainer"]["valid_fold"]
    in_channels = config_params["trainer"]["in_channels"]
    num_classes = config_params["trainer"]["classes"]
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
    lr = config_params["trainer"]["learning_rate"]
    batch_size = config_params["trainer"]["batch_size"]
    window_size = config_params["trainer"]["window_size"]
    weight_func = config_params["model"]["weights"]
    loss_func = config_params["model"]['loss_function']

    weights_path = os.path.join(root_path, f"script/MODEL_LAYER/{weight_func}_weights_{which_label}.csv")

    print(f"Train for log {log_num}")

    # LOGGING

    logger.add(os.path.join(root_path, f'script/experiments/{config_params["trainer"]["log_num"]}/info.log'))

    log_path = os.path.join(root_path,f'script/experiments/{log_num}')
    # writer = SummaryWriter(f'./experiments/{log_num}/tensorboard')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(log_path, 'checkpoints'), exist_ok=True)

    shutil.copy(config_path, os.path.join(log_path, 'config.yml'))
    shutil.copy(os.path.join(root_path, f'script/MODEL_LAYER/dataset.py'), os.path.join(log_path, f'dataset.py'))
    shutil.copy(os.path.join(root_path,f'script/MODEL_LAYER/train.py'), os.path.join(log_path, f'train.py'))

    # MODEL PARAMS

    ENCODER = config_params['model']['encoder']
    # ENCODER_WEIGHTS = 'imagenet' #None
    # ACTIVATION = None  # could be None for logits (binary) or 'softmax2d' for multicalss segmentation

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = SMPUnet(encoder_name=ENCODER, in_channels=15, num_classes=6)
    # print(model)

    fold_data_path = os.path.join(config_params['data_loader']['train_paths'], f'{sample_mode}_sampling_{which_label}_5_fold')

    train_set = MLFluvDataset(
        data_path=fold_data_path,
        mode='train',
        folds=train_fold,
        window=window_size,
        label=which_label,
        one_hot_encode=False
    )

    val_set = MLFluvDataset(
        data_path=fold_data_path,
        mode='val',
        folds=valid_fold,
        window=window_size, 
        label=which_label,
        one_hot_encode=False
    )

    # Use saved weights for loss function, if the weights are pre-calculated 
    if os.path.isfile(weights_path):
        df = pd.read_csv(weights_path)
        class_weights = df['Weights']
    else:
        print('Going to calculate weight now..')
        class_weights = get_class_weight(train_set, weight_func=weight_func,suffix=which_label)
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
            force_reload=False)

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
        log_num=log_num
    )
    
    interface.train(epochs=epochs, eval_interval=10)
