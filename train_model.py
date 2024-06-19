import os
import subprocess
import yaml
from sklearn.model_selection import ParameterGrid
from UTILS import utils

script_path = 'script/'
script_path, is_vm = utils.update_root_path_for_machine(root_path=script_path)

if is_vm:
    config_path = os.path.join(script_path,'config.yml')
else:
    config_path = os.path.join(script_path, 'config_k8s.yml')

# Split data into 5 folds for train, validation 
# When change labels among ESRI, DW, ESAWC, only re-run this step
# subprocess.run(['python', os.path.join(script_path,'DATA_LAYER/split_data.py'), '--config_path', config_path])

# Train initial UNet model with a given label from existing LULC maps
# subprocess.run(['python', os.path.join(script_path,'MODEL_LAYER/train.py', '--config_path'), config_path])
# Make predition on the trained model
# subprocess.run(['python', 'os.path.join(script_path,'MODEL_LAYER/inference.py'), '--config_path', config_path])
# Fine tune the model by add a new class (sediment), using new data for training as well.

# Grid search to tune hyperparameters for incremental learning
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Define the grid for temperature and distill_lamda
param_grid = {'temperature': [1, 0.1, 0.5, 2, 10], 'distill_lamda': [0, 0.25, 0.5, 0.75, 1]}

# Create the parameter grid
grid = ParameterGrid(param_grid)

for params in grid:
    print(params)
    # Update the config
    config['incremental_learning']['temperature'] = params['temperature']
    config['incremental_learning']['distill_lamda'] = params['distill_lamda']

    # Increment the tune_log_num
    config['incremental_learning']['tune_log_num'] += 1
    
    # Save the config back to yaml
    with open('script/config.yml', 'w') as f:
        yaml.safe_dump(config, f)
    
    subprocess.run(['python', os.path.join(script_path, 'MODEL_LAYER/fine_tune.py'), '--config_path', config_path])