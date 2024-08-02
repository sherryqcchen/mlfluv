import os
import subprocess
import numpy as np
import yaml
from sklearn.model_selection import ParameterGrid
from UTILS import utils

script_path = 'script/'
script_path, is_vm = utils.update_root_path_for_machine(root_path=script_path)
print(script_path)

if is_vm:
    config_path = os.path.join(script_path,'config.yml')
else:
    config_path = os.path.join(script_path, 'config_k8s.yml')

# Split data into 5 folds for train, validation 
# When change labels among ESRI, DW, ESAWC, only re-run this step
# subprocess.run(['python', os.path.join(script_path,'DATA_LAYER/split_data.py'), '--config_path', config_path])

# Train initial UNet model with a given label from existing LULC maps
# subprocess.run(['python', os.path.join(script_path,'MODEL_LAYER/train.py'), '--config_path', config_path])

# Make predition on the trained model
# subprocess.run(['python', os.path.join(script_path,'MODEL_LAYER/inference.py'), '--config_path', config_path])

# Fine tune the model by add a new class (sediment), using new data for training as well.

# Grid search to tune hyperparameters for incremental learning
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

temperature_list = np.arange(1,11)
lambda_list = np.arange(0, 1.25, 0.25)

# Define the grid for temperature and distill_lamda
param_grid = {'temperature': temperature_list, 'distill_lamda': lambda_list, 'freeze_encoder': [True, False]}

# Create the parameter grid
grid = ParameterGrid(param_grid)

# Start the tune log number from 1 for the first tune 
start_tune_number = 0

for params in grid:
    print(params)
    # Update the config
    config['incremental_learning']['temperature'] = int(params['temperature'])
    config['incremental_learning']['distill_lamda'] = float(params['distill_lamda'])
    config['incremental_learning']['freeze_encoder'] = params['freeze_encoder']

    # Increment the tune_log_num
    start_tune_number += 1
    config['incremental_learning']['tune_log_num'] = start_tune_number
    
    # Save the config back to yaml
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    
    subprocess.run(['python', os.path.join(script_path, 'MODEL_LAYER/incremental.py'), '--config_path', config_path])