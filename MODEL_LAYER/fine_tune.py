# this script is for incremental learning 
import csv
import json
import torch.nn as nn
import torch
import torch.optim as optim
from loguru import logger
import numpy as np
import os
import copy

from model import SMPUnet
from dataset import MLFluvDataset
from interface import MLFluvUnetInterface
from weight_calculator import get_class_weight
# from utils import parse_config_params

def parse_config_params(config_file):
    with open(config_file, 'r') as f:
        config_params = json.load(f)
    return config_params

if __name__ == "__main__":

    exp_folder = './experiments/100'
    # output_folder = os.path.join(exp_folder, 'preds')
    # os.makedirs(output_folder, exist_ok=True)

    SHOW_PLOTS = False

    # TODO Add logger functions and save refined model

    config_params = parse_config_params(os.path.join(exp_folder, 'config.json'))

    log_num = config_params["trainer"]["log_num"]
    in_channels = config_params["trainer"]["in_channels"]
    classes = 7 # config_params["trainer"]["classes"]
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
    lr = config_params["trainer"]["learning_rate"]
    loss_func = config_params["model"]['loss_function']
    batch_size = 2 # config_params["trainer"]["batch_size"]
    weight_func = config_params["model"]["weights"]
    window_size = config_params["trainer"]["window_size"]
    temperature = config_params["model"]['temperature']
    weights_path = config_params["model"]['weights_path']
    freeze_encoder = config_params["model"]['freeze_encoder']

    ENCODER = config_params['model']['encoder']
    ENCODER_WEIGHTS = None
    ACTIVATION = None
    
    # create an untrained model, with one extra class in num_classes
    old_net = SMPUnet(encoder_name="resnet34", in_channels=15, num_classes=classes, num_valid_classes=6, encoder_freeze=freeze_encoder, temperature=0.5)
    print(f"{old_net.temperature=}")
    print()

    train_set = MLFluvDataset(
        config_params['data_loader']['args']['data_paths'],
        mode='train',
        label='hand',
        folds = [0, 1, 2, 3],
        one_hot_encode=False      
    )

    val_set = MLFluvDataset(
        config_params['data_loader']['args']['data_paths'],
        mode='val',
        label='hand',
        folds = [4],
        one_hot_encode=False      
    )
    
    # load pretrain model weights
    checkpoint_path = os.path.join(exp_folder, 'checkpoints', os.listdir(os.path.join(exp_folder, 'checkpoints'))[0])
    old_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    old_net.eval()
    
    # Create new UNet by copying the old Unet
    new_net = copy.deepcopy(old_net)
    new_net.num_valid_classes = 7
    print(f"{new_net.temperature=}")

    # Use saved weights for loss function, if the weights are pre-calculated 
    if os.path.isfile(weights_path):
        class_weights = list(csv.reader(open(weights_path, "r"), delimiter=","))
        class_weights = np.array([float(i) for i in class_weights[0]])
    else:
        class_weights = get_class_weight(train_set, weight_func=weight_func)
    print(class_weights)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # SET LOSS, OPTIMIZER
    if loss_func == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(reduction='mean',
                                        weight=weights,
                                        label_smoothing=0.01) 
                                        # ignore_index=0)
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

    optimiser = optim.Adam(new_net.parameters(), lr=lr)

    interface = MLFluvUnetInterface(
        model=new_net,
        data_train=train_set,
        data_val=val_set,
        loss_fn=criterion,
        optimiser=optimiser,
        device=device,
        batch_size=batch_size,
        log_num=log_num,
        distill_lamda=0.25,
        old_model=old_net
    )
    
    interface.train(epochs=epochs, eval_interval=5)
