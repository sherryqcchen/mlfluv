import cv2
import torch
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import os
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import MLFluvDataset
from utils import parse_config_params, extract_patches, reconstruct_from_patches
import matplotlib.pyplot as plt
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def infer_with_patches(img, net, config_params, preprocess_fn=None):
    log_num = config_params["trainer"]["log_num"]
    device = config_params["trainer"]["device"]
    num_classes = config_params["trainer"]["classes"]
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    # checkpoint_path = f'./experiments/{log_num}/checkpoints/best_model.pth'

    # save_root_folder = f'./experiments/{log_num}/predictions'
    # os.makedirs(save_root_folder, exist_ok=True)

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
    exp_folder = './experiments/2'
    output_folder = os.path.join(exp_folder, 'preds')
    os.makedirs(output_folder, exist_ok=True)

    SHOW_PLOTS = False

    config_params = parse_config_params(os.path.join(exp_folder, 'config.json'))

    log_num = config_params["trainer"]["log_num"]
    in_channels = config_params["trainer"]["in_channels"]
    classes = config_params["trainer"]["classes"]
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
    lr = config_params["trainer"]["learning_rate"]
    batch_size = config_params["trainer"]["batch_size"]
    window_size = config_params["trainer"]["window_size"]

    # LOGGING

    logger.add(f'experiments/{config_params["trainer"]["log_num"]}/preds.log')

    ENCODER = config_params['model']['encoder']
    ENCODER_WEIGHTS = None
    ACTIVATION = None

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)

    model = smp.Unet(encoder_name=ENCODER,
                     encoder_weights=ENCODER_WEIGHTS,
                     in_channels=in_channels,
                     decoder_attention_type='scse',
                     classes=classes,
                     activation=ACTIVATION
                     ).to(device)

    test_set = MLFluvDataset(
        config_params['data_loader']['args']['data_paths'],
        mode='test',
        folds = [0, 1, 2, 3, 4]        
    )

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)  # TODO: workers

    checkpoint_path = os.path.join(exp_folder, 'checkpoints', os.listdir(os.path.join(exp_folder, 'checkpoints'))[0])
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    for i, (image, mask) in enumerate(test_loader):
        image, mask = image.to(device), mask.to(device)
        
        if int(window_size) == 512:
            y_pred = model(image).cpu().detach().numpy()
        else:
            # Inference with patches, because the data tile size is not the same as window size
            y_pred = infer_with_patches(np.transpose(image.cpu().detach().numpy()[0, :, :], (1, 2, 0)), model, config_params)

        # y_pred_prob = model(image)

        # # plot true-color S2 image
        # true_color = cv2.normalize(np.transpose(image.cpu().numpy()[0, 3:2:1, :, :], (1 , 2, 0)),
        #                            dst=None,
        #                            alpha=0,
        #                            beta=255,
        #                            norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        # # plot VV S1
        # vv_img = cv2.normalize(image.cpu().numpy()[0, -2, :, :],
        #                        dst=None,
        #                        alpha=0,
        #                        beta=255,
        #                        norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        y_pred_map = torch.from_numpy(y_pred).argmax(dim=0)
        print(np.unique(y_pred_map.numpy()))

        y = mask.cpu().detach().numpy()[0, :, :]
        print(np.unique(y))

        if SHOW_PLOTS:
            # plt.imshow(np.where(y_pred > .9, 1, 0))
            # plt.imshow(y_pred)
            # plt.show()

            plt.imshow(true_color)
            plt.show()
            #
            plt.imshow(vv_img)
            plt.show()

            plt.imshow(y)
            plt.show()

        cv2.imwrite(os.path.join(output_folder, f'{i}_mask.png'), (y * 255).astype(np.uint8))
        # cv2.imwrite(os.path.join(output_folder, f'{i}_true_color_img.png'), true_color)
        # cv2.imwrite(os.path.join(output_folder, f'{i}_vv_img.png'), vv_img)
        cv2.imwrite(os.path.join(output_folder, f'{i}_pred.png'), (y_pred_map.numpy() * 255).astype(np.uint8))
        
        tp, fp, fn, tn = smp.metrics.get_stats(y_pred_map, mask.cpu().squeeze().long(), mode='multiclass', num_classes=classes)
        # compute metric
        test_micro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro") # TODO find out which reduction is a correct usage
        test_macro_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        test_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        test_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        test_accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        test_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

        logger.info(f"Testing)")
        logger.info(f"{'':<10}Micro IOU{'':<1} ----> {round(test_micro_iou.item(), 3)}")
        logger.info(f"{'':<10}Macro IOU{'':<1} ----> {round(test_macro_iou.item(), 3)}")
        logger.info(f"{'':<10}Accuracy{'':<1} ----> {round(test_accuracy.item(), 3)}")
        logger.info(f"{'':<10}Recall{'':<1} ----> {round(test_recall.item(), 3)}")
        logger.info(f"{'':<10}Precision{'':<1} ----> {round(test_precision.item(), 3)}")
        logger.info(f"{'':<10}F1{'':<1} ----> {round(test_f1.item(), 3)}")