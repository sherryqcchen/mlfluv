import cv2
import json
from loguru import logger
import matplotlib.pyplot as plt
import os
import shutil
import sklearn
# from torchmetrics.functional import accuracy, precision, recall, f1_score, jaccard_index  # , iou
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall, JaccardIndex, F1Score
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as utils
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

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


    # TODO get weights based on the pixel count of each class in train set 
    # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
    
    def get_class_weight(dataset, weight_func='inverse_log'):
        labels = [label for _, label in dataset]

        train_labels = np.stack(labels, axis=0).flatten()
        # print(train_labels.shape)
        pixel_sum = train_labels.shape[0]
        classes, frequencies = np.unique(train_labels, return_counts=True)
        class_percent = frequencies / pixel_sum
        # print(classes)
        # print(frequencies)
        # print(np.bincount(train_labels))

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

    # weights = get_class_weight(train_set, weight_func='inverse_log')
    # print(weights)
    # weights_count = get_class_weight(train_set, weight_func='inverse_count')
    # print(weights_count)
    
    class_weights = get_class_weight(train_set, weight_func='sklearn')
    print(class_weights)

    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    
    # SET LOSS, OPTIMIZER

    # criterion = nn.BCEWithLogitsLoss(reduction='sum')  # changed from nn.BCELoss
    # TODO change reduction to 'none' causing error, find out which one I should use
    criterion = nn.CrossEntropyLoss(reduction='sum',
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

    acc_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    precision_metric = Precision(task="multiclass", num_classes=num_classes).to(device)
    recall_metric = Recall(task='multiclass', num_classes=num_classes).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=num_classes).to(device)
    cm_metric = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)
    iou_metric = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)



    for epoch in range(1, epochs + 1):

        model.train()

        train_loss = 0  # summation of loss for every batch
        train_acc = 0  # summation of accuracy for every batch
        train_recall = 0
        train_precision = 0
        train_f1 = 0
        train_iou = 0
        train_cm = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
            optimizer.zero_grad()

            y_pred = model(X_batch)


            # Convert with argmac to shape the output n_classes layers to only one layer.
            # https://github.com/qubvel/segmentation_models.pytorch/issues/541
            # y_pred = torch.argmax(y_pred, dim =1).float()

            # y_pred_prob = torch.sigmoid(y_pred)
            
            # TODO the following line works for one_hot_encode=True 
            loss = criterion(y_pred, y_batch.argmax(dim=1))
            # loss = criterion(y_pred, y_batch)
            

            # y_pred_prob = torch.sigmoid(y_pred) # sigmoid for binary segmentation: https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/
            train_loss += loss.item()

            train_acc = acc_metric(y_pred, y_batch)
            train_precision = precision_metric(y_pred, y_batch)
            train_recall = recall_metric(y_pred, y_batch)
            train_cm = cm_metric(y_pred, y_batch)
            train_iou = iou_metric(y_pred, y_batch)
            train_f1 = f1_metric(y_pred, y_batch)

            loss.backward()
            optimizer.step()

        train_acc = acc_metric.compute()
        train_precision = precision_metric.compute()
        train_recall = recall_metric.compute()
        train_cm = cm_metric.compute()
        train_iou = iou_metric.compute()
        train_f1 = f1_metric.compute()

        logger.info(f"EPOCH: {epoch} (training)")
        logger.info(f"{'':<10}Loss{'':<5} ----> {train_loss / len(train_set):.3f}")
        logger.info(f"{'':<10}Accuracy{'':<1} ----> {round(train_acc.item(), 3)}")
        logger.info(f"{'':<10}Recall{'':<1} ----> {round(train_recall.item(), 3)}")
        logger.info(f"{'':<10}Precision{'':<1} ----> {round(train_precision.item(), 3)}")
        logger.info(f"{'':<10}F1{'':<1} ----> {round(train_f1.item(), 3)}")
        logger.info(f"{'':<10}Mean IOU{'':<1} ----> {round(train_iou.item(), 3)}")
        logger.info(f"{'':<10}Confusion Matrix{'':<1}\n{train_cm}")

        # if epoch % 5 == 0:  # if the number of epoch is divided by 5 do the validation

        acc_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        cm_metric.reset()
        iou_metric.reset()
        f1_metric.reset()

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
            # y_val_pred_prob = torch.sigmoid(y_val_pred)
            val_loss += loss.item()

            val_acc = acc_metric(y_val_pred_prob, y_batch.long())
            val_precision = precision_metric(y_val_pred_prob, y_batch.long())
            val_recall = recall_metric(y_val_pred_prob, y_batch.long())
            val_cm = cm_metric(y_val_pred_prob, y_batch.long())
            val_iou = iou_metric(y_val_pred_prob, y_batch.long())
            val_f1 = f1_metric(y_val_pred_prob, y_batch.long())

        val_acc = acc_metric.compute()
        val_precision = precision_metric.compute()
        val_recall = recall_metric.compute()
        val_cm = cm_metric.compute()
        val_iou = iou_metric.compute()
        val_f1 = f1_metric.compute()

        if val_f1.item() >= best_val_miou:
            best_val_miou = val_f1.item()
            best_val_epoch = epoch
            torch.save(model.state_dict(), f'./experiments/{log_num}/checkpoints/best_model.pth')
            logger.info(f'\n\nSaved new model at epoch {epoch}!\n\n')

        logger.info(f"EPOCH: {epoch} (validation)")
        logger.info(f"{'':<10}Loss{'':<5} ----> {val_loss / len(val_set)}")
        logger.info(f"{'':<10}Accuracy{'':<1} ----> {round(val_acc.item(), 3)}")
        logger.info(f"{'':<10}Recall{'':<1} ----> {round(val_recall.item(), 3)}")
        logger.info(f"{'':<10}Precision{'':<1} ----> {round(val_precision.item(), 3)}")
        logger.info(f"{'':<10}F1{'':<1} ----> {round(val_f1.item(), 3)}")
        logger.info(f"{'':<10}IOU{'':<1} ----> {round(val_iou.item(), 3)}")
        logger.info(f"{'':<10}Confusion Matrix{'':<1}\n{val_cm}")

        acc_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        cm_metric.reset()
        iou_metric.reset()
        f1_metric.reset()

    logger.info(f'Best Flood IoU: {best_val_miou} at epoch {best_val_epoch}')

