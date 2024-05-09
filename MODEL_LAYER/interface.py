import shutil
import time
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn as nn
import torch
import os
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

class MLFluvUnetInterface():
    """
    Train interface for semantic segmentation on MLFluv dataset.
 

    Args:
    model: a choice of U-Net model.
    data_train (DataLoader): prepared torch tensor of train dataset.
    data_val (DataLoader): prepared torch tensor of train dataset.
    loss_fn (nn.Module): A choice of loss function, for example torch.nn.MSELoss().
    optimiser (torch.optim.Optimiser): A chice of optimiser, e.g., torch.optim.Adam().
    device (str): choose from "cuda" and "cpu".
    batch_size: batch_size (int): The batch size used for training.
    log_dir (str): a path string pointing to the location of the log.

    Returns:
    loss: the loss value.
    pred: prediction of a time sequence value.
    y: the actual value from a future day of the time sequence. 

    Example:
    >>> MLFluvUnetInterface(
            model,
            dataset_train,
            dataset_val,
            loss_fn,
            optimiser,
            device,
            batch_size=BATCH_SIZE,
            log_num=log_num
        )

    """   

    def __init__(
        self,
        model,
        data_train,
        data_val,
        loss_fn,
        optimiser,
        device,
        batch_size,
        log_num,
        mode='initial_train',
        distill_lamda=0,
        old_model=None,
    ):
        self.device = device
        self.model = model.to(device)
        self.log_num = log_num
        self.mode = mode

        if self.mode == "initial_train":
            self.best_model_path = f'./experiments/{self.log_num}/checkpoints/best_model.pth'
        else:
            self.best_model_path = f'./experiments/{self.log_num}/{self.mode}/checkpoints/best_model.pth'

        self.old_model = old_model

        if self.old_model is not None:
            self.old_model = old_model.to(device)
            # Freeze old model
            for param in self.old_model.parameters():
                param.requires_grad = False
 
        self.data_train = data_train
        self.data_val = data_val

        self.criterion = loss_fn
        self.optimiser = optimiser
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, mode='min', factor=0.1, patience=10)

        self.batch_size = batch_size
        self.num_classes = self.model.num_classes
        self.distill_lamda = distill_lamda

        self.dataloader_train = DataLoader(self.data_train,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=2) # TODO: remove num_workers when debugging
        self.dataloader_val = DataLoader(self.data_val,
                                         batch_size=self.batch_size,
                                         shuffle=False, 
                                         num_workers=2) # TODO: remove num_workers when debugging

        self.writer =  SummaryWriter()    



        self.losses_train = []
        self.losses_val = []

        self.best_val_miou = 0
        self.best_val_epoch = 0

        if self.log_num is not None:
            # LOGGING
            logger.add(f'experiments/{self.log_num}/info.log')

            os.makedirs(f'./experiments/{log_num}', exist_ok=True)
            os.makedirs(f'./experiments/{log_num}/checkpoints', exist_ok=True)

            shutil.copy('MODEL_LAYER/config.json', os.path.join(f'./experiments/{log_num}', 'config.json'))
            shutil.copy(f'MODEL_LAYER/dataset.py', os.path.join(f'./experiments/{log_num}', f'dataset.py'))
            shutil.copy(f'MODEL_LAYER/train.py', os.path.join(f'./experiments/{log_num}', f'train.py'))

    def train_1epoch(self, epoch_idx):
        # train on one epoch and calculate train loss
        t_start = time.time()

        train_loss = 0  # summation of loss for every batch
        train_recall = 0
        train_precision = 0
        train_jaccard_index = JaccardIndex(task='multiclass', num_classes=self.num_classes, ignore_index=0, average='none').to(self.device)
        
        self.model.train()
        for batch_idx, (X_batch, y_batch) in enumerate(self.dataloader_train):

            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        
            self.optimiser.zero_grad()

            y_pred = self.model(X_batch)

            y_pred_softmax = nn.functional.softmax(y_pred, dim=1)

            loss_ce = self.criterion(y_pred_softmax, y_batch)

            if self.old_model is not None:
                # Distillation loss
                y_old = self.old_model(X_batch) # [batch_size, num_classes, image_height, image_width]
                mask = y_batch < self.old_model.num_valid_classes # [batch_size, 1, image_height, image_width]
                mask = torch.broadcast_to(mask.unsqueeze(1), (mask.shape[0], self.num_classes, mask.shape[1], mask.shape[2]))
                
                y_old[torch.logical_not(mask)] = -1e20 # -torch.inf triggers exponentiation operation overflows, so we use -1e20 instead
                probabilities_old = nn.functional.softmax(y_old, dim=1) # <--- old probablitliy

                y_new = self.old_model.apply_mask(y_pred, self.old_model.num_valid_classes)
                y_new = y_new / self.model.temperature 
                probabilities_new = nn.functional.softmax(y_new, dim=1) # < --- new probability

                loss_distill = self.criterion(probabilities_old, probabilities_new)
                loss_total = (1-self.distill_lamda) * loss_ce + self.distill_lamda * loss_distill
            else:
                loss_total = loss_ce

            train_loss += loss_total.item()

            # Convert with argmax to reshape the output n_classes layers to only one layer.
            y_pred_agx = y_pred_softmax.argmax(dim=1) 

            tp, fp, fn, tn = smp.metrics.get_stats(y_pred_agx, y_batch, mode='multiclass', num_classes=self.num_classes)
            # compute metric
            train_micro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro") # TODO find out which reduction is a correct usage
            train_macro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
            train_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            train_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

            train_jaccard_index.update(y_pred_agx, y_batch)

            loss_total.backward()
            self.optimiser.step()
        
        self.losses_train.append(train_loss / len(self.data_train))
        self.writer.add_scalar('Loss/train', train_loss / len(self.data_train), epoch_idx)

        train_ious = train_jaccard_index.compute()

        # Compute class-wise IoU from the Jaccard Index
        class_wise_iou_train = []
        for class_idx in range(self.num_classes):
            class_iou = train_ious[class_idx]
            class_wise_iou_train.append(class_iou.item())
        
        # Compute mean IoU across all classes
        train_miou = sum(class_wise_iou_train) / len(class_wise_iou_train)

        logger.info(f"EPOCH: {epoch_idx} (training)")
        logger.info(f"{'':<10}Loss{'':<5} ----> {train_loss / len(self.data_train):.3f}")
        logger.info(f"{'':<10}Mean IoU{'':<1} ----> {round(train_miou, 3)}")
        logger.info(f"{'':<10}Class-wise IoU{'':<1} ----> {class_wise_iou_train}")
        logger.info(f"{'':<10}Micro IoU{'':<1} ----> {round(train_micro_iou.item(), 3)}")
        logger.info(f"{'':<10}Macro IoU{'':<1} ----> {round(train_macro_iou.item(), 3)}")
        logger.info(f"{'':<10}Recall{'':<1} ----> {round(train_recall.item(), 3)}")
        logger.info(f"{'':<10}Precision{'':<1} ----> {round(train_precision.item(), 3)}")

        train_jaccard_index.reset()

        t_end = time.time()
        total_time = t_end - t_start
        print("Train epoch time : {:.1f}s".format(total_time))

    def eval(self, dataloader, epoch_idx):

        val_jaccard_index = JaccardIndex(task='multiclass', num_classes=self.num_classes, ignore_index=0, average='none').to(self.device)
        self.model.eval()

        val_loss = 0
        val_recall = 0
        val_precision = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)  # send values to device (GPU)
                y_val_pred = self.model(X_batch)
                y_val_pred_softmax = nn.functional.softmax(y_val_pred, dim=1)

                loss_ce = self.criterion(y_val_pred_softmax, y_batch)

                if self.old_model is not None:
                    # Distillation loss
                    y_old = self.old_model(X_batch) # [batch_size, num_classes, image_height, image_width]
                    mask = y_batch < self.old_model.num_valid_classes # [batch_size, 1, image_height, image_width]
                    mask = torch.broadcast_to(mask.unsqueeze(1), (mask.shape[0], self.num_classes, mask.shape[1], mask.shape[2]))
                    
                    y_old[torch.logical_not(mask)] = -1e20 #-torch.inf
                    probabilities_old = nn.functional.softmax(y_old, dim=1) # <--- old probablitliy

                    y_new = self.old_model.apply_mask(y_val_pred, self.old_model.num_valid_classes)
                    y_new = y_new / self.model.temperature
                    probabilities_new = nn.functional.softmax(y_new, dim=1) # < --- new probability

                    loss_distill = self.criterion(probabilities_old, probabilities_new)
                    
                    loss_total = (1-self.distill_lamda) * loss_ce + self.distill_lamda * loss_distill
                else:
                    loss_total = loss_ce

                val_loss += loss_total.item()

                y_val_pred_agx = y_val_pred_softmax.argmax(dim=1)

                tp, fp, fn, tn = smp.metrics.get_stats(y_val_pred_agx, y_batch, mode='multiclass', num_classes=self.num_classes)

                val_micro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro") # TODO find out which reduction is a correct usage
                val_macro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
                val_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
                val_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")        

                val_jaccard_index.update(y_val_pred_agx, y_batch)

        self.scheduler.step(val_loss)
        print("lr:", self.scheduler._last_lr)
        
        self.losses_val.append(val_loss / len(self.data_val))
        self.writer.add_scalar('Loss/val', val_loss / len(self.data_val), epoch_idx)

        val_ious = val_jaccard_index.compute()

        # Compute class-wise IoU from the Jaccard Index
        class_wise_iou_val = []
        for class_idx in range(self.num_classes):
            class_iou = val_ious[class_idx]
            class_wise_iou_val.append(class_iou.item())
        
        # Compute mean IoU across all classes
        val_miou = sum(class_wise_iou_val) / len(class_wise_iou_val)

        # if val_miou.item() >= best_val_miou:
        if val_miou >= self.best_val_miou:
            self.best_val_miou = val_miou
            self.best_val_epoch = epoch_idx
            torch.save(self.model.model.state_dict(), self.best_model_path)
            logger.info(f'\n\nSaved new model at epoch {epoch_idx}!\n\n')

        logger.info(f"EPOCH: {epoch_idx} (validating)")
        logger.info(f"{'':<10}Loss{'':<5} ----> {val_loss / len(self.data_val):.3f}")
        logger.info(f"{'':<10}Mean IoU{'':<1} ----> {round(val_miou, 3)}")
        logger.info(f"{'':<10}Class-wise IoU{'':<1} ----> {class_wise_iou_val}")
        logger.info(f"{'':<10}Micro IOU{'':<1} ----> {round(val_micro_iou.item(), 3)}")
        logger.info(f"{'':<10}Macro IOU{'':<1} ----> {round(val_macro_iou.item(), 3)}")
        logger.info(f"{'':<10}Recall{'':<1} ----> {round(val_recall.item(), 3)}")
        logger.info(f"{'':<10}Precision{'':<1} ----> {round(val_precision.item(), 3)}")

        val_jaccard_index.reset()  

        return
                

    def train(self, epochs, eval_interval=1):

        for epoch in range(epochs):
                
            self.train_1epoch(epoch)
            if epoch % eval_interval == 0:
                self.eval(self.dataloader_val,epoch)
                # TODO how to arrage logging functions to be here
                logger.info(f'Best mean IoU: {self.best_val_miou} at epoch {self.best_val_epoch}')
            self.writer.close()


        
