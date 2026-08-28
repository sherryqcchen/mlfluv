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
from torch.utils.data import Subset

from model import SMPSegmentationModel
from dataset import MLFluvDataset
from interface import MLFluvUnetInterface
from UTILS import utils
from weight_calculator import get_class_weight

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

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
    s1_bands = config_params.get("sample", {}).get("s1_bands", []) or []
    s2_bands = config_params.get("sample", {}).get("s2_bands", []) or []

    bands = s1_bands + s2_bands  # This will work even if one of them is missing
    print(bands)
    in_channels = len(bands) # config_params["trainer"]["in_channels"]

    data_loader_cfg = config_params["data_loader"]
    which_label = data_loader_cfg["which_label"]
    nan_handling = data_loader_cfg.get("nan_handling", "mask")
    s2_source = data_loader_cfg.get("s2_source", "main")

    log_num = config_params["trainer"]["log_num"]
    train_fold = config_params["trainer"]["train_fold"]
    valid_fold = config_params["trainer"]["valid_fold"]
    batch_size = config_params["trainer"]["batch_size"]
    num_workers = config_params["trainer"].get("num_workers", 2)
    max_train_samples = config_params["trainer"].get("max_train_samples")
    max_val_samples = config_params["trainer"].get("max_val_samples")
    num_classes = config_params["trainer"]["classes"]
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
    eval_interval = config_params["trainer"].get("eval_interval", 10)
    scheduler_config = config_params["trainer"].get("scheduler", {})
    optimizer_config = config_params["trainer"].get("optimizer", {})
    lr = config_params["trainer"]["learning_rate"]
    window_size = config_params["trainer"]["window_size"]
    patch_size = config_params["sample"]["patch_size"]
    exp_folder = config_params["trainer"].get("exp_folder", os.path.join(SCRIPT_ROOT, "experiments"))
    if not os.path.isabs(exp_folder):
        exp_folder = os.path.join(SCRIPT_ROOT, exp_folder)
    weight_func = config_params["model"]["weights"]
    loss_func = config_params["model"]['loss_function']
    configured_class_weights = config_params["model"].get("class_weights")

    weights_path = os.path.join(SCRIPT_ROOT, f"MODEL_LAYER/{weight_func}_weights_{which_label}.csv")

    print(f"Train for log {log_num}")

    log_path = os.path.join(exp_folder, f'{log_num}')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(log_path, 'checkpoints'), exist_ok=True)

    # LOGGING
    logger.add(os.path.join(log_path, 'info.log'))
    # writer = SummaryWriter(f'./experiments/{log_num}/tensorboard')

    shutil.copy(args.config_path, os.path.join(log_path, 'config.yml'))
    shutil.copy(os.path.join(SCRIPT_ROOT, f'MODEL_LAYER/dataset.py'), os.path.join(log_path, f'dataset.py'))
    shutil.copy(os.path.join(SCRIPT_ROOT, f'MODEL_LAYER/train.py'), os.path.join(log_path, f'train.py'))

    # MODEL PARAMS

    model_cfg = config_params['model']
    ARCHITECTURE = model_cfg.get('architecture', 'Unet')
    ENCODER = model_cfg['encoder']
    ENCODER_WEIGHTS = model_cfg['encoder_weights'] #None
    # ACTIVATION = None  # could be None for logits (binary) or 'softmax2d' for multicalss segmentation

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = SMPSegmentationModel(
        architecture=ARCHITECTURE,
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=in_channels,
        num_classes=num_classes,
        num_valid_classes=num_classes,
        decoder_attention_type=model_cfg.get('decoder_attention_type', 'scse'),
        encoder_depth=model_cfg.get('encoder_depth', 5),
        encoder_output_stride=model_cfg.get('encoder_output_stride', 16),
        decoder_channels=model_cfg.get('decoder_channels', 256),
        decoder_atrous_rates=tuple(model_cfg.get('decoder_atrous_rates', (12, 24, 36))),
        decoder_aspp_separable=model_cfg.get('decoder_aspp_separable', True),
        decoder_aspp_dropout=model_cfg.get('decoder_aspp_dropout', 0.5),
        decoder_segmentation_channels=model_cfg.get('decoder_segmentation_channels', 256),
        upsampling=model_cfg.get('upsampling', 4),
        aux_params=model_cfg.get('aux_params'),
    )
    # print(model)

    fold_dir_name = f'{sample_mode}_sampling_{which_label}_5_fold'
    fold_output_suffix = data_loader_cfg.get("fold_output_suffix")
    if fold_output_suffix:
        fold_dir_name = f"{fold_dir_name}_{fold_output_suffix}"
    fold_data_path = data_loader_cfg.get("fold_data_dir")
    if fold_data_path is None:
        fold_data_path = os.path.join(data_loader_cfg['train_paths'], fold_dir_name)
    elif not os.path.isabs(fold_data_path):
        fold_data_path = os.path.join(SCRIPT_ROOT, fold_data_path)
    print(f"Using fold data path: {fold_data_path}")

    train_set = MLFluvDataset(
        data_path=fold_data_path,
        mode='initial_train',
        folds=train_fold,
        window_size=window_size,
        patch_size=patch_size,
        label=which_label,
        one_hot_encode=False,
        bands=bands,
        nan_handling=nan_handling,
        s2_source=s2_source
    )

    val_set = MLFluvDataset(
        data_path=fold_data_path,
        mode='initial_train',
        folds=valid_fold,
        window_size=window_size,
        patch_size=patch_size,
        label=which_label,
        one_hot_encode=False,
        bands=bands,
        nan_handling=nan_handling,
        s2_source=s2_source
    )

    if max_train_samples is not None:
        train_set = Subset(train_set, range(min(int(max_train_samples), len(train_set))))
        print(f"Using first {len(train_set)} training samples for this run.")
    if max_val_samples is not None:
        val_set = Subset(val_set, range(min(int(max_val_samples), len(val_set))))
        print(f"Using first {len(val_set)} validation samples for this run.")

    # Use saved weights for loss function, if the weights are pre-calculated 
    if configured_class_weights is not None:
        class_weights = configured_class_weights
        print("Using class weights from config.")
    elif os.path.isfile(weights_path):
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
    
    optimizer_name = optimizer_config.get("name", "Adam")
    weight_decay = optimizer_config.get("weight_decay", 0)
    if optimizer_name == "Adam":
        optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    interface = MLFluvUnetInterface(
        model=model,
        data_train=train_set,
        data_val=val_set,
        loss_fn=criterion,
        optimiser=optimiser,
        device=device,
        batch_size=batch_size,
        log_num=log_num,
        num_workers=num_workers,
        run_config_path=args.config_path,
        exp_folder=exp_folder,
        scheduler_config=scheduler_config
    )
    
    interface.train(epochs=epochs, eval_interval=eval_interval)
