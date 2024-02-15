import cv2
import json
from loguru import logger
import matplotlib.pyplot as plt
import os
import shutil
import sklearn
import segmentation_models_pytorch as smp
import torch

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
# from stream_metrics import StreamSegMetrics

from dataset import MLFluvDataset


def parse_config_params(config_file):
    with open(config_file, 'r') as f:
        config_params = json.load(f)
    return config_params


if __name__ == "__main__":

    ####################################
    # PARSE CONFIG FILE
    ####################################
    print("test for log 1")
    config_params = parse_config_params('config.json')

    log_num = config_params["trainer"]["log_num"]
    in_channels = config_params["trainer"]["in_channels"]
    num_classes = config_params["trainer"]["classes"]
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
    lr = config_params["trainer"]["learning_rate"]
    batch_size = config_params["trainer"]["batch_size"]

# LOGGING

    logger.add(f'experiments/{config_params["trainer"]["log_num"]}/info.log')
    # writer = SummaryWriter(f'./experiments/{log_num}/tensorboard')

    os.makedirs(f'./experiments/{log_num}', exist_ok=True)
    os.makedirs(f'./experiments/{log_num}/checkpoints', exist_ok=True)

    shutil.copy('config.json', os.path.join(f'./experiments/{log_num}', 'config.json'))
    shutil.copy(f'dataset.py', os.path.join(f'./experiments/{log_num}', f'dataset.py'))
    shutil.copy(f'train.py', os.path.join(f'./experiments/{log_num}', f'train.py'))

    # tb_logger = TensorboardLogger(log_dir=f'./experiments/{log_num}/tensorboard')

    # MODEL PARAMS

    ENCODER = config_params['model']['encoder']
    ENCODER_WEIGHTS = None
    ACTIVATION = 'softmax2d'# None  # could be None for logits (binary) or 'softmax2d' for multicalss segmentation

    device = 'cpu'#torch.device(device if torch.cuda.is_available() else "cpu")

    model = smp.Unet(encoder_name=ENCODER,
                     encoder_weights=ENCODER_WEIGHTS,
                     in_channels=in_channels,
                     decoder_attention_type='scse',
                     classes=num_classes,
                     activation=ACTIVATION
                     ).to(device)
    # print(model)

    train_set = MLFluvDataset(
        data_path=config_params['data_loader']['args']['data_paths'],
        mode='train',
        folds = [0, 1, 2, 3],
        window=256,
        label='hand',
        one_hot_encode=False
    )

    val_set = MLFluvDataset(
        data_path=config_params['data_loader']['args']['data_paths'],
        mode='train',
        folds = [4],
        window=256, 
        label='hand',
        one_hot_encode=False
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # , num_workers=2) # TODO: remove num_workers when debugging
    val_loader = DataLoader(val_set, batch_size=1) #, num_workers=2) # TODO: remove num_workers when debugging



    
    def get_class_weight(dataset, weight_func='inverse_log'):
        # get weights based on the pixel count of each class in train set 
        # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
        labels = [label for _, label in dataset]

        train_labels = np.stack(labels, axis=0).flatten()

        pixel_sum = train_labels.shape[0]
        classes, frequencies = np.unique(train_labels, return_counts=True)
        class_percent = frequencies / pixel_sum

        if weight_func == 'inverse_log':
            # weight = 1 / np.log(class_percent)
            weight = pixel_sum / (len(classes) * np.log(frequencies.astype(np.float64)))
        elif weight_func == 'inverse_count': 
            weight = pixel_sum / (len(classes) * frequencies.astype(np.float64))
        elif weight_func == 'inverse_sqrt':
            weight = pixel_sum / (len(classes) * np.sqrt(class_percent.astype(np.float64)))
        else: 
            print('No weight function is given. We use sklearn compute class weight function')
            # weight = np.ones(classes.shape[0], dtype=np.float64)
            weight = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)

        # the weight for class 7 (clouds and no data) is not needed, so it should be zero out
        if 7 in classes:
            weight[-1] = 0

        return weight
    
    class_weights = get_class_weight(train_set, weight_func='inverse_log')
    print(class_weights)

    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # SET LOSS, OPTIMIZER
    # TODO change reduction to 'none' causing error, find out which one I should use
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    weight=weights,
                                    label_smoothing=0.25)
    # criterion = smp.losses.DiceLoss(mode='multiclass')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_acc': [],
        'train_recall': [],
        'train_precision': [],
        'train_f1': [],
        'train_cm': [],
        'train_iou': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_cm': [],
        'val_iou': [],
    }

    best_val_miou = 0
    best_val_epoch = 0
    

    # metric = StreamSegMetrics(num_classes)
    
    for epoch in range(1, epochs + 1):

        model.train()

        train_loss = 0  # summation of loss for every batch
        train_acc = 0  # summation of accuracy for every batch
        train_recall = 0
        train_precision = 0
        train_f1 = 0
        train_iou = 0
        train_cm = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            train_loss += loss.item()

            # Convert with argmax to reshape the output n_classes layers to only one layer.
            y_pred = y_pred.argmax(dim=1) 

            tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y_batch, mode='multiclass', num_classes=num_classes)
            # compute metric
            train_micro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro") # TODO find out which reduction is a correct usage
            train_macro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
            train_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            train_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            train_accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
            train_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

            # metric.update(y_batch.cpu().numpy(), y_pred.cpu().numpy())
            # metrics_out = metric.get_results()
            # print(metrics_out)

            loss.backward()
            optimizer.step()
        
        logger.info(f"EPOCH: {epoch} (training)")
        logger.info(f"{'':<10}Loss{'':<5} ----> {train_loss / len(train_set):.3f}")
        logger.info(f"{'':<10}Micro IOU{'':<1} ----> {round(train_micro_iou.item(), 3)}")
        logger.info(f"{'':<10}Macro IOU{'':<1} ----> {round(train_macro_iou.item(), 3)}")
        logger.info(f"{'':<10}Accuracy{'':<1} ----> {round(train_accuracy.item(), 3)}")
        logger.info(f"{'':<10}Recall{'':<1} ----> {round(train_recall.item(), 3)}")
        logger.info(f"{'':<10}Precision{'':<1} ----> {round(train_precision.item(), 3)}")
        logger.info(f"{'':<10}F1{'':<1} ----> {round(train_f1.item(), 3)}")
        # logger.info(f"{'':<10}Mean IOU{'':<1} ----> {round(train_iou.item(), 3)}")
        # logger.info(f"{'':<10}Confusion Matrix{'':<1}\n{train_cm}")

       

        model.eval()

        val_loss = 0
        val_acc = 0
        val_recall = 0
        val_precision = 0
        val_f1 = 0
        val_iou = 0
        val_cm = 0

        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # send values to device (GPU)
            y_val_pred = model(X_batch)

            # y_val_pred_np = (torch.argmax(y_val_pred, dim=1).cpu().numpy()[0, :, :]) * 120
            # cv2.imwrite(os.path.join(f'./experiments/{log_num}/val_{i}.png'), y_val_pred_np.astype(np.uint8))

            loss = criterion(y_val_pred, y_batch)
            val_loss += loss.item()
            # Convert with argmax to reshape the output n_classes layers to only one layer.
            y_val_pred = y_val_pred.argmax(dim=1) 

            # metric.update(y_batch.cpu().numpy(), y_val_pred.cpu().numpy())
            # metrics_out = metric.get_results()
            # print(metrics_out)

            tp, fp, fn, tn = smp.metrics.get_stats(y_val_pred, y_batch, mode='multiclass', num_classes=num_classes)
            # compute metric
            val_micro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro") # TODO find out which reduction is a correct usage
            val_macro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
            val_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            val_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            val_accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
            val_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")        

        if val_micro_iou.item() >= best_val_miou:
            best_val_miou = val_micro_iou.item()
            best_val_epoch = epoch
            torch.save(model.state_dict(), f'./experiments/{log_num}/checkpoints/best_model.pth')
            logger.info(f'\n\nSaved new model at epoch {epoch}!\n\n')

        logger.info(f"EPOCH: {epoch} (validating)")
        logger.info(f"{'':<10}Loss{'':<5} ----> {val_loss / len(val_set):.3f}")
        logger.info(f"{'':<10}Micro IOU{'':<1} ----> {round(val_micro_iou.item(), 3)}")
        logger.info(f"{'':<10}Macro IOU{'':<1} ----> {round(val_macro_iou.item(), 3)}")
        logger.info(f"{'':<10}Accuracy{'':<1} ----> {round(val_accuracy.item(), 3)}")
        logger.info(f"{'':<10}Recall{'':<1} ----> {round(val_recall.item(), 3)}")
        logger.info(f"{'':<10}Precision{'':<1} ----> {round(val_precision.item(), 3)}")
        logger.info(f"{'':<10}F1{'':<1} ----> {round(train_f1.item(), 3)}")
        # logger.info(f"{'':<10}Confusion Matrix{'':<1}\n{val_cm}")


    logger.info(f'Best micro IoU: {best_val_miou} at epoch {best_val_epoch}')

