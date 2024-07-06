# this script is for incremental learning 
import argparse
import cv2
import shutil
import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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
from UTILS import utils
from UTILS.utils import load_config
from UTILS.plotter import plot_inference_result


if __name__ == "__main__":

    root_path = ''
    root_path, is_vm = utils.update_root_path_for_machine(root_path=root_path)

    final_tune_path = 'script/experiments/2003/fine_tune_15'
    config_path = os.path.join(final_tune_path, 'config.yml')
    
    config_params = load_config(config_path)
    
    sample_mode = config_params["sample"]["sample_mode"]
    which_label = config_params["data_loader"]["which_label"]
    log_num = config_params["trainer"]["log_num"]
    train_fold = config_params["trainer"]["train_fold"]
    valid_fold = config_params["trainer"]["valid_fold"]
    in_channels = config_params["trainer"]["in_channels"]
    classes = config_params["trainer"]["classes"] + 1 # 7
    device = config_params["trainer"]["device"]
    epochs = 200 # config_params["trainer"]["epochs"]
    lr = config_params["trainer"]["learning_rate"]
    loss_func = config_params["model"]['loss_function']
    batch_size = config_params["trainer"]["batch_size"]
    weight_func = config_params["model"]["weights"]
    window_size = config_params["trainer"]["window_size"]
    with_extra_urban = config_params["incremental_learning"]['with_extra_urban']
    temperature = config_params["incremental_learning"]['temperature']
    distill_lamda = config_params["incremental_learning"]['distill_lamda']
    freeze_encoder = config_params["incremental_learning"]['freeze_encoder']
    tune_mode = f'fine_tune_{config_params["incremental_learning"]["tune_log_num"]}'
    ENCODER = config_params['model']['encoder']
    ENCODER_WEIGHTS = None
    ACTIVATION = None

    exp_folder = os.path.join(root_path, f'script/experiments/final_tune')
    output_folder = os.path.join(root_path, f'script/experiments/final_tune/{log_num}/{tune_mode}')
    os.makedirs(output_folder, exist_ok=True)

    SHOW_PLOTS = False

    weights_path = os.path.join(root_path, f"script/MODEL_LAYER/{weight_func}_weights_{which_label}_final.csv")

    print(f'Final tune for incremental model {tune_mode} from log {log_num}')
    print(f"{temperature = }")
    print(f"{distill_lamda = }")

    # Logging
    logger.add(os.path.join(output_folder, 'info.log'))

    os.makedirs(os.path.join(output_folder, 'checkpoints'), exist_ok=True)
    shutil.copy(os.path.join(root_path, f'script/MODEL_LAYER/final_tune.py'), os.path.join(output_folder, 'final_tune.py'))
    shutil.copy(config_path, os.path.join(output_folder, 'config.yml'))

    # create an untrained model, with one extra class in num_classes
    model = SMPUnet(encoder_name="resnet34", in_channels=15, num_classes=classes, num_valid_classes=7, encoder_freeze=freeze_encoder, temperature=temperature)
    print(f"{model.temperature=}")

    checkpoint_path = os.path.join(final_tune_path, 'checkpoints', os.listdir(os.path.join(final_tune_path, 'checkpoints'))[0])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    train_set = MLFluvDataset(
        data_path=os.path.join(config_params['data_loader']['train_paths'], f'final_test_{which_label}_4_fold'),
        mode='train',
        label='hand',
        folds=train_fold,
        one_hot_encode=False      
    )

    val_set = MLFluvDataset(
        data_path=os.path.join(config_params['data_loader']['train_paths'], f'final_test_{which_label}_fold'),
        mode='val',
        label='hand',
        folds=[0],
        one_hot_encode=False      
    )

    # Use saved weights for loss function, if the weights are pre-calculated 
    if os.path.isfile(weights_path):
        df = pd.read_csv(weights_path)
        class_weights = df['Weights']
    else:
        class_weights = get_class_weight(train_set, weight_func=weight_func, suffix=f'{which_label}_final')
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
        mode='final_tune',
        distill_lamda=distill_lamda,
        best_model_path=os.path.join(output_folder, 'checkpoints/best_model.pth')
    )
    
    interface.train(epochs=epochs, eval_interval=5)

    # Making predictions

    # Do inference in CPU, overwrite device with 'cpu'
    device = 'cpu'

    pred_folder = os.path.join(output_folder, 'preds')
    os.makedirs(pred_folder, exist_ok=True)
    
    # LOGGING
    logger.add(os.path.join(output_folder,'preds.log'))

    test_set = MLFluvDataset(
        data_path=os.path.join(config_params['data_loader']['train_paths'], f'final_test_{which_label}_fold'),
        mode='test',
        label='hand',
        folds=None,
        one_hot_encode=False      
    )

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)  # TODO: workers

    best_final_checkpoint_path = os.path.join(output_folder, 'checkpoints', os.listdir(os.path.join(output_folder, 'checkpoints'))[0])

    final_net = model.to(device)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    final_net.load_state_dict(torch.load(best_final_checkpoint_path, map_location=device))
    final_net.eval()

    # calculate IoU
    test_jaccard_index = JaccardIndex(task='multiclass', num_classes=classes, ignore_index=0, average='none').to(device)

    # Initialize accumulators for accuracy metrics
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_tp_per_class = [0] * classes
    total_fp_per_class = [0] * classes
    total_fn_per_class = [0] * classes
    total_tn_per_class = [0] * classes
    # Initialize a list to hold the sum of IoU for each class across all images
    sum_iou_per_class = [0] * classes

    mIoUs = []
    class_IoUs = []

    for i, (image, mask) in enumerate(test_loader):
        image, mask = image.to(device), mask.to(device)
        
        if int(window_size) == 512:
            y_pred = final_net(image).cpu().detach().numpy().squeeze()
        else:
            # Inference with patches, because the data tile size is not the same as window size
            y_pred = infer_with_patches(np.transpose(image.cpu().detach().numpy()[0, :, :], (1, 2, 0)), final_net, config_params)

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

        # Accumulate stats per class
        for class_idx in range(classes):
            total_tp_per_class[class_idx] += tp[class_idx]
            total_fp_per_class[class_idx] += fp[class_idx]
            total_fn_per_class[class_idx] += fn[class_idx]
            total_tn_per_class[class_idx] += tn[class_idx]

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
        logger.info(f"{'':<10}Class wise IoU{'':<1} ----> {class_wise_iou_test}")
        logger.info(f"{'':<10}Micro IOU{'':<1} ----> {round(test_micro_iou.item(), 3)}")
        logger.info(f"{'':<10}Macro IOU{'':<1} ----> {round(test_macro_iou.item(), 3)}")
        logger.info(f"{'':<10}Accuracy{'':<1} ----> {round(test_accuracy.item(), 3)}")
        logger.info(f"{'':<10}Recall{'':<1} ----> {round(test_recall.item(), 3)}")
        logger.info(f"{'':<10}Precision{'':<1} ----> {round(test_precision.item(), 3)}")
        logger.info(f"{'':<10}F1{'':<1} ----> {round(test_f1.item(), 3)}")
        
        # Accumulate stats
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        mIoUs.append(test_miou)
        class_IoUs.append(class_wise_iou_test)

    # We use mIoU calculated by mean (sum (mIoU of each image)), which is more sensitive to the accuracy of minority classes
    test_miou_overall = sum(mIoUs)/len(mIoUs)

    # Calculate the mean class-wise IoU for all class-wise IoU per image
    class_iou_overall = np.mean(np.array(class_IoUs), axis=0).tolist()

    # Compute metrics using total stats
    test_micro_iou_overall = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction="micro")
    test_macro_iou_overall = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction="macro")
    test_f1_overall = smp.metrics.f1_score(total_tp, total_fp, total_fn, total_tn, reduction="micro")
    test_precision_overall = smp.metrics.precision(total_tp, total_fp, total_fn, total_tn, reduction="micro")
    test_accuracy_overall = smp.metrics.accuracy(total_tp, total_fp, total_fn, total_tn, reduction="micro")
    test_recall_overall = smp.metrics.recall(total_tp, total_fp, total_fn, total_tn, reduction="micro")

    logger.info(f"Overall Testing Result)")
    logger.info(f"{'':<10}Mean IOU{'':<1} ----> {round(test_miou_overall, 3)}")
    logger.info(f"{'':<10}Micro IOU{'':<1} ----> {round(test_micro_iou_overall.item(), 3)}")
    logger.info(f"{'':<10}Macro IOU{'':<1} ----> {round(test_macro_iou_overall.item(), 3)}")
    logger.info(f"{'':<10}Accuracy{'':<1} ----> {round(test_accuracy_overall.item(), 3)}")
    logger.info(f"{'':<10}Recall{'':<1} ----> {round(test_recall_overall.item(), 3)}")
    logger.info(f"{'':<10}Precision{'':<1} ----> {round(test_precision_overall.item(), 3)}")
    logger.info(f"{'':<10}F1{'':<1} ----> {round(test_f1_overall.item(), 3)}")
    logger.info(f"{'':<10}Class wise IoU{'':<1} ----> {class_iou_overall}")