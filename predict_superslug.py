
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from UTILS.utils import load_config
from UTILS.plotter import plot_inference_result
from MODEL_LAYER.dataset import MLFluvDataset
from MODEL_LAYER.model import SMPUnet
from MODEL_LAYER.inference import infer_with_patches
from DATA_LAYER.split_data import get_s12label_list, split_n_folds 


DATA_PREPARED = True

if not DATA_PREPARED:
    clean_data_path = 'data/clean_data/superslug_s12dw_data_clean_MANUAL'

    predict_list = get_s12label_list('DW', clean_data_path)

    print(len(predict_list))

    split_n_folds(1, predict_list, save_dir='data/fold_data/superslug_predict_1_fold', which_label='DW')


# Load pre-trained best incremental model from MLFLUV 
final_tune_path = 'script/experiments/final_tune/2003/fine_tune_34'
config_path = os.path.join(final_tune_path, 'config.yml')
config_params = load_config(config_path)
classes = config_params["trainer"]["classes"] + 1 # 7
freeze_encoder = config_params["incremental_learning"]['freeze_encoder']
print(classes)
print(freeze_encoder)
temperature = 1
distill_lamda = 0

device = 'cpu'

checkpoint_path = os.path.join(final_tune_path, 'checkpoints', os.listdir(os.path.join(final_tune_path, 'checkpoints'))[0])
model = SMPUnet(encoder_name="resnet34", in_channels=15, num_classes=classes, num_valid_classes=7, encoder_freeze=freeze_encoder, temperature=temperature)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

pred_set = MLFluvDataset(
    data_path = 'data/fold_data/superslug_predict_1_fold',
    mode = 'test',
    label = 'DW',
    folds = None,
    one_hot_encode = False      
)


test_loader = DataLoader(pred_set, batch_size=1, shuffle=False)


for i, (image, mask) in enumerate(test_loader):
    image, mask = image.to(device), mask.to(device)
    print(image.shape)
    
    if int(image.shape[-1]) == 512:
        y_pred = model(image).cpu().detach().numpy().squeeze()
    else:
        # Inference with patches, because the data tile size is not the same as window size
        y_pred = infer_with_patches(np.transpose(image.cpu().detach().numpy()[0, :, :], (1, 2, 0)), model, config_params)

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
    
    plot_inference_result(s2_rgb, s1_vv, y, y_pred_map, 'data/predict_data', i)
    
    