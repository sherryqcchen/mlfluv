# this script is for incremental learning 
import cv2
import json
import shutil
import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex

from model import SMPUnet
from dataset import MLFluvDataset
from interface import MLFluvUnetInterface
from weight_calculator import get_class_weight
from inference import infer_with_patches
from UTILS.utils import parse_config_params
from UTILS.plotter import plot_inference_result

# from utils import parse_config_params

def parse_config_params(config_file):
    with open(config_file, 'r') as f:
        config_params = json.load(f)
    return config_params

if __name__ == "__main__":
    tune_mode = 'fine_tune_01'
    IF_PREDICT = True

    exp_folder = './experiments/1001'
    output_folder =f'./experiments/1001/{tune_mode}'
    os.makedirs(output_folder, exist_ok=True)

    SHOW_PLOTS = False

    config_params = parse_config_params(os.path.join(exp_folder, 'config.json'))
    
    log_num = config_params["trainer"]["log_num"]
    in_channels = config_params["trainer"]["in_channels"]
    classes = 7 
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
    lr = config_params["trainer"]["learning_rate"]
    loss_func = config_params["model"]['loss_function']
    batch_size = 2 # config_params["trainer"]["batch_size"]
    weight_func = config_params["model"]["weights"]
    window_size = config_params["trainer"]["window_size"]
    temperature = config_params["model"]['temperature']
    freeze_encoder = config_params["model"]['freeze_encoder']

    ENCODER = config_params['model']['encoder']
    ENCODER_WEIGHTS = None
    ACTIVATION = None

    weights_path = f"MODEL_LAYER/{weight_func}_weights_hand.csv"

    print(f'Fine tune for log {log_num}')

    # Logging
    logger.add(os.path.join(output_folder, 'info.log'))

    os.makedirs(os.path.join(output_folder, 'checkpoints'), exist_ok=True)
    shutil.copy(f'MODEL_LAYER/fine_tune.py', os.path.join(output_folder, 'fine_tune.py'))
    shutil.copy('MODEL_LAYER/config.json', os.path.join(output_folder, 'config.json'))

    # create an untrained model, with one extra class in num_classes
    old_net = SMPUnet(encoder_name="resnet34", in_channels=15, num_classes=classes, num_valid_classes=6, encoder_freeze=freeze_encoder, temperature=0.5)
    print(f"{old_net.temperature=}")

    train_set = MLFluvDataset(
        config_params['data_loader']['args']['tune_paths'],
        mode='train',
        label='hand',
        folds = [0, 1, 2, 3],
        one_hot_encode=False      
    )

    val_set = MLFluvDataset(
        config_params['data_loader']['args']['tune_paths'],
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
        df = pd.read_csv(weights_path)
        class_weights = df['Weights']
    else:
        class_weights = get_class_weight(train_set, weight_func=weight_func, suffix='hand')
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
        mode=tune_mode,
        distill_lamda=0.25,
        old_model=old_net
    )
    
    interface.train(epochs=epochs, eval_interval=5)

    # Making predictions
    if IF_PREDICT is True:
        # Do inference in CPU, overwrite device with 'cpu'
        device = 'cpu'

        pred_folder = os.path.join(output_folder, 'preds')
        os.makedirs(pred_folder, exist_ok=True)
        
        # LOGGING
        logger.add(os.path.join(output_folder,'preds.log'))

        test_set = MLFluvDataset(
            config_params['data_loader']['args']['tune_paths'],
            mode='test',
            label='hand',
            folds = [4],
            one_hot_encode=False      
        )
    
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)  # TODO: workers

        checkpoint_path = os.path.join(output_folder, 'checkpoints', os.listdir(os.path.join(exp_folder, 'checkpoints'))[0])

        new_net = new_net.to(device)
        # model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
        new_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        new_net.eval()

        # calculate IoU
        test_jaccard_index = JaccardIndex(task='multiclass', num_classes=classes, ignore_index=0, average='none').to(device)

        for i, (image, mask) in enumerate(test_loader):
            image, mask = image.to(device), mask.to(device)
            
            if int(window_size) == 512:
                y_pred = new_net(image).cpu().detach().numpy().squeeze()
            else:
                # Inference with patches, because the data tile size is not the same as window size
                y_pred = infer_with_patches(np.transpose(image.cpu().detach().numpy()[0, :, :], (1, 2, 0)), new_net, config_params)

            y_pred_map = torch.from_numpy(y_pred).argmax(dim=0).cpu()
            print(f"The classes show in the prediction: {np.unique(y_pred_map.numpy())}")

            y = mask.cpu().detach().numpy()[0, :, :]
            print(f"The classes show in the label: {np.unique(y)}")
            
            # Plot the S2 rgb, S1 vv, maks and prediction
            s2_rgb = cv2.normalize(np.transpose(image.cpu().numpy()[0, 5:2:-1, :, :], (1,2,0)),
                                    dst=None,
                                    alpha=0,
                                    beta=255,
                                    norm_type=cv2.NORM_MINMAX).astype(np.uint8)
            s1_vv = image.cpu().numpy()[0,0,:,:]
            
            plot_inference_result(s2_rgb, s1_vv, y, y_pred_map, os.path.join(output_folder,'preds'), i)
            
            tp, fp, fn, tn = smp.metrics.get_stats(y_pred_map, mask.cpu().squeeze().long(), mode='multiclass', num_classes=classes)
            # compute metric
            test_micro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro") # TODO find out which reduction is a correct usage
            test_macro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
            test_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            test_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            test_accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
            test_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

            test_jaccard_index.update(y_pred_map, mask.cpu().squeeze().long())
            test_ious = test_jaccard_index.compute()

            # Compute class-wise IoU from the Jaccard Index
            class_wise_iou_test = []
            for class_idx in range(classes):
                class_iou = test_ious[class_idx]
                class_wise_iou_test.append(class_iou.item())
            
            # Compute mean IoU across all classes
            test_miou = sum(class_wise_iou_test) / len(class_wise_iou_test)

            logger.info(f"Testing)")
            logger.info(f"{'':<10}Mean IOU{'':<1} ----> {round(test_miou, 3)}")
            logger.info(f"{'':<10}Class-wise IoU{'':<1} ----> {class_wise_iou_test}")
            logger.info(f"{'':<10}Micro IOU{'':<1} ----> {round(test_micro_iou.item(), 3)}")
            logger.info(f"{'':<10}Macro IOU{'':<1} ----> {round(test_macro_iou.item(), 3)}")
            logger.info(f"{'':<10}Accuracy{'':<1} ----> {round(test_accuracy.item(), 3)}")
            logger.info(f"{'':<10}Recall{'':<1} ----> {round(test_recall.item(), 3)}")
            logger.info(f"{'':<10}Precision{'':<1} ----> {round(test_precision.item(), 3)}")
            logger.info(f"{'':<10}F1{'':<1} ----> {round(test_f1.item(), 3)}")