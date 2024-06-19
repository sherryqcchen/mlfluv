import subprocess
import os
from UTILS import utils

script_path = 'script/'
script_path, is_vm = utils.update_root_path_for_machine(root_path=script_path)

if is_vm:
    config_path = os.path.join(script_path,'config.yml')
else:
    config_path = os.path.join(script_path, 'config_k8s.yml')

# Sample points from HydroSHEDS network 
# subprocess.run(['python', os.path.join(script_path,'DATA_LAYER/random_sampler.py'), '--config_path', config_path])

# Call the downloading script to get data from EE API
# subprocess.run(['python', os.path.join(script_path,'DATA_LAYER/get_ee_data.py'), '--config_path', config_path])

# Call the data preprocess script
# subprocess.run(['python', os.path.join(script_path,'DATA_LAYER/label_prepare.py'), '--config_path', config_path])

# Split data into 5 folds for train, validation 
# When change labels among ESRI, DW, ESAWC, only re-run this step
subprocess.run(['python', os.path.join(script_path,'DATA_LAYER/split_data.py'), '--config_path', config_path])