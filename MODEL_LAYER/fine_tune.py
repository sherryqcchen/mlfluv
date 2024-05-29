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
from UTILS.utils import load_config
from UTILS.plotter import plot_inference_result


if __name__ == "__main__":
    ####################################
    # PARSE CONFIG FILE
    ####################################
    parser = argparse.ArgumentParser(description="Please provide a configuration ymal file for trainning a U-Net model.")
    parser.add_argument('--config_path',type=str, default='script/config.yml', help='Path to a configuration yaml file.' )
    parser.add_argument('--if_predict',type=bool,default=True, help='True if make prediction using fine-tuned model.')
    args = parser.parse_args()

    config_params = load_config(args.config_path)
    
    sample_mode = config_params["sample"]["sample_mode"]
    which_label = config_params["data_loader"]["which_label"]
    log_num = config_params["trainer"]["log_num"]
    train_fold = config_params["trainer"]["train_fold"]
    valid_fold = config_params["trainer"]["valid_fold"]
    in_channels = config_params["trainer"]["in_channels"]
    classes = config_params["trainer"]["classes"] + 1 # 7
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
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

    exp_folder = f'script/experiments/{log_num}'
    output_folder =f'script/experiments/{log_num}/{tune_mode}'
    os.makedirs(output_folder, exist_ok=True)

    SHOW_PLOTS = False

    weights_path = f"script/MODEL_LAYER/{weight_func}_weights_{which_label}_fintune.csv"

    print(f'Fine tune {tune_mode} for log {log_num}')
    print(f"{temperature = }")
    print(f"{distill_lamda = }")

    # Logging
    logger.add(os.path.join(output_folder, 'info.log'))

    os.makedirs(os.path.join(output_folder, 'checkpoints'), exist_ok=True)
    shutil.copy(f'script/MODEL_LAYER/fine_tune.py', os.path.join(output_folder, 'fine_tune.py'))
    shutil.copy('script/config.yml', os.path.join(output_folder, 'config.yml'))

    # create an untrained model, with one extra class in num_classes
    old_net = SMPUnet(encoder_name="resnet34", in_channels=15, num_classes=classes, num_valid_classes=6, encoder_freeze=freeze_encoder, temperature=temperature)
    print(f"{old_net.temperature=}")

    if with_extra_urban:
        fold_data_path = os.path.join(config_params['data_loader']['train_paths'], f'finetune_with_urban_{which_label}_5_fold')
    else:
        fold_data_path = os.path.join(config_params['data_loader']['train_paths'], f'finetune_{which_label}_5_fold')
    train_set = MLFluvDataset(
        data_path=fold_data_path,
        mode='train',
        label=which_label,
        folds=train_fold,
        one_hot_encode=False      
    )

    val_set = MLFluvDataset(
        data_path=fold_data_path,
        mode='val',
        label=which_label,
        folds=valid_fold,
        one_hot_encode=False      
    )
    
    # load pretrain model weights
    checkpoint_path = os.path.join(exp_folder, 'checkpoints', os.listdir(os.path.join(exp_folder, 'checkpoints'))[0])
    old_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    old_net.eval()
    
    # Create new UNet by copying the old Unet
    new_net = copy.deepcopy(old_net)
    new_net.num_valid_classes = classes
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
        distill_lamda=distill_lamda,
        old_model=old_net
    )
    
    # interface.train(epochs=epochs, eval_interval=5)

    # Making predictions
    if args.if_predict is True:
        # Do inference in CPU, overwrite device with 'cpu'
        device = 'cpu'

        pred_folder = os.path.join(output_folder, 'preds')
        os.makedirs(pred_folder, exist_ok=True)
        
        # LOGGING
        logger.add(os.path.join(output_folder,'preds.log'))

        test_set = MLFluvDataset(
            data_path=f'data/fold_data/test_{which_label}_fold',
            mode='test',
            label='hand',
            folds = None,
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

        # Initialize accumulators for accuracy metrics
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_tp_per_class = [0] * classes
        total_fp_per_class = [0] * classes
        total_fn_per_class = [0] * classes
        total_tn_per_class = [0] * classes
        # Initialize a list to hold the sum of IoU for each class across all images
        sum_iou_per_class = [0] * classes

        mIoUs = []

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
            logger.info(f"{'':<10}Class-wise IoU{'':<1} ----> {class_wise_iou_test}")
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

        # # Compute class-wise IoU using total stats per class for the whole test dataset
        # class_wise_iou_test = []
        # for class_idx in range(classes):
        #     class_iou = smp.metrics.iou_score(
        #         total_tp_per_class[class_idx], 
        #         total_fp_per_class[class_idx], 
        #         total_fn_per_class[class_idx], 
        #         total_tn_per_class[class_idx], 
        #         reduction="none")
            
        #     class_wise_iou_test.append(class_iou.tolist())
        
        # # Sum the IoU for each class across all images
        # for iou_list in class_wise_iou_test:
        #     for class_idx in range(classes):
        #         sum_iou_per_class[class_idx] += iou_list[class_idx]

        # Compute mean IoU across all classes
        # The following mIoU calculates sum (intersection of all images) / sum (union of all images)
        # This mIoU is higher and more biased to the majority class
        # test_miou_overall = sum(sum_iou_per_class) / (len(class_wise_iou_test) * classes)
        
        # class_wise_iou_overall = [i/classes for i in sum_iou_per_class]

        # We use mIoU calculated by mean (sum (mIoU of each image)), which is more sensitive to the accuracy of minority classes
        test_miou_overall = sum(mIoUs)/len(mIoUs)


        # Compute metrics using total stats
        test_micro_iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction="micro")
        test_macro_iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction="macro")
        test_f1 = smp.metrics.f1_score(total_tp, total_fp, total_fn, total_tn, reduction="micro")
        test_precision = smp.metrics.precision(total_tp, total_fp, total_fn, total_tn, reduction="micro")
        test_accuracy = smp.metrics.accuracy(total_tp, total_fp, total_fn, total_tn, reduction="micro")
        test_recall = smp.metrics.recall(total_tp, total_fp, total_fn, total_tn, reduction="micro")

        logger.info(f"Overall Testing Result)")
        logger.info(f"{'':<10}Mean IOU{'':<1} ----> {round(test_miou_overall, 3)}")
        # logger.info(f"{'':<10}Class-wise IoU{'':<1} ----> {class_wise_iou_test}")
        logger.info(f"{'':<10}Micro IOU{'':<1} ----> {round(test_micro_iou.item(), 3)}")
        logger.info(f"{'':<10}Macro IOU{'':<1} ----> {round(test_macro_iou.item(), 3)}")
        logger.info(f"{'':<10}Accuracy{'':<1} ----> {round(test_accuracy.item(), 3)}")
        logger.info(f"{'':<10}Recall{'':<1} ----> {round(test_recall.item(), 3)}")
        logger.info(f"{'':<10}Precision{'':<1} ----> {round(test_precision.item(), 3)}")
        logger.info(f"{'':<10}F1{'':<1} ----> {round(test_f1.item(), 3)}")