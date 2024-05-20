import subprocess

# Train initial UNet model with a given label from existing LULC maps
subprocess.run(['python', 'script/MODEL_LAYER/train.py', '--config_path', 'script/config.yml'])
# Make predition on the trained model
subprocess.run(['python', 'script/MODEL_LAYER/inference.py', '--config_path', 'script/config.yml'])
# Fine tune the model by add a new class (sediment), using new data for training as well.
subprocess.run(['python', 'script/MODEL_LAYER/fine_tune.py', '--config_path', 'script/config.yml'])