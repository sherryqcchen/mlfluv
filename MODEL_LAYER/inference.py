import argparse
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import cv2
import torch
from loguru import logger
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex

from dataset import MLFluvDataset
from UTILS import utils
from UTILS.utils import load_config, extract_patches, reconstruct_from_patches
from UTILS.plotter import plot_inference_result


def infer_with_patches(img, net, config_params, preprocess_fn=None):
    log_num = config_params["trainer"]["log_num"]
    device = config_params["trainer"]["device"]
    num_classes = config_params["trainer"]["classes"]
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
 
    net.eval()

    h, w, _ = img.shape
    if preprocess_fn is not None:
        img = preprocess_fn(img)

    patches, count_occ_map, hw_combs, padded_img_shape = extract_patches(img, patch_size=(256, 256),
                                                                         stride=(128, 128))
    
    prob_patches = []

    for patch in tqdm(patches):

        patch = np.array(patch).astype(np.float32)

        if len(patch.shape) == 1:
            patch = np.expand_dims(patch, 0)
        else:
            patch = np.transpose(patch, (2, 0, 1))
        X = torch.from_numpy(patch).unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(X)
            if output.size()[1] <= 2:
                probs = torch.sigmoid(output)
            else:
                probs = torch.softmax(output, dim=1)

        probs_as_np = probs.squeeze().cpu().numpy()

        prob_patches.append(probs_as_np)

    mask_shape = (img.shape[0], img.shape[1])
    padded_mask_shape = (padded_img_shape[0], padded_img_shape[1])
    # probs = reconstruct_from_patches_with_clip(mask_shape, prob_patches, hw_combs, padded_mask_shape)
    probs = reconstruct_from_patches(mask_shape, prob_patches, count_occ_map, hw_combs, padded_mask_shape, classes=num_classes)
    return probs

if __name__ == '__main__':

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
    config_params = load_config(args.config_path)

    log_num = config_params["trainer"]["log_num"]

    # load the config file again from saved experiments

    exp_folder = os.path.join(root_path, f'script/experiments/{log_num}')
    output_folder = os.path.join(exp_folder, 'preds')
    os.makedirs(output_folder, exist_ok=True)

    SHOW_PLOTS = False

    config_params = load_config(os.path.join(exp_folder, 'config.yml'))

    log_num = config_params["trainer"]["log_num"]
    in_channels = config_params["trainer"]["in_channels"]
    classes = config_params["trainer"]["classes"]
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
    lr = config_params["trainer"]["learning_rate"]
    batch_size = config_params["trainer"]["batch_size"]
    window_size = config_params["trainer"]["window_size"]
    which_label = config_params['data_loader']['which_label']

    # LOGGING

    logger.add(os.path.join(root_path,f'script/experiments/{log_num}/preds.log'))

    ENCODER = config_params['model']['encoder']
    ENCODER_WEIGHTS = None
    ACTIVATION = None

    device = 'cpu' #torch.device(device if torch.cuda.is_available() else "cpu")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)

    model = smp.Unet(encoder_name=ENCODER,
                     encoder_weights=ENCODER_WEIGHTS,
                     in_channels=in_channels,
                     decoder_attention_type='scse',
                     classes=classes,
                     activation=ACTIVATION
                     ).to(device)

    test_set = MLFluvDataset(
        data_path=os.path.join(root_path,f'data/fold_data/test_{which_label}_fold'),
        mode='test',
        label=which_label,
        folds = None,
        one_hot_encode=False      
    )
    # print(test_set.num_classes)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)  # TODO: workers

    checkpoint_path = os.path.join(exp_folder, 'checkpoints', os.listdir(os.path.join(exp_folder, 'checkpoints'))[0])
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

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
            y_pred = model(image).cpu().detach().numpy().squeeze()
        else:
            # Inference with patches, because the data tile size is not the same as window size
            y_pred = infer_with_patches(np.transpose(image.cpu().detach().numpy()[0, :, :], (1, 2, 0)), model, config_params)

        y_pred_map = torch.from_numpy(y_pred).argmax(dim=0)
        print(f"Pred unique: {np.unique(y_pred_map.numpy())}")

        y = mask.cpu().detach().numpy()[0, :, :]
        print(f"Label unique: {np.unique(y)}")
        
        # ESRI label for the initial train has very few 6 in it. Convert them to 5
        if 6 in y:
            y[y==6] = 5
            print("Found it and fixed it.")
            print(f"Fixed Label unique: {np.unique(y)}")
        
        # Plot the S2 rgb, S1 vv, maks and prediction
        s2_rgb = cv2.normalize(np.transpose(image.numpy()[0, 5:2:-1, :, :], (1,2,0)),
                                   dst=None,
                                   alpha=0,
                                   beta=255,
                                   norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        s1_vv = image.numpy()[0,0,:,:]
        
        plot_inference_result(s2_rgb, s1_vv, y, y_pred_map, output_folder, i)
        
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
