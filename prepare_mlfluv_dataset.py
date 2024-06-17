import subprocess

# Sample points from HydroSHEDS network 
# subprocess.run(['python', 'script/DATA_LAYER/random_sampler.py', '--config_path', 'script/config.yml'])

# Call the downloading script to get data from EE API
# subprocess.run(['python', 'script/DATA_LAYER/get_ee_data.py', '--config_path', 'script/config.yml'])

# Call the data preprocess script
# subprocess.run(['python', 'script/DATA_LAYER/label_prepare.py', '--config_path', 'script/config.yml'])

# Split data into 5 folds for train, validation 
# When change labels among ESRI, DW, ESAWC, only re-run this step
subprocess.run(['python', 'script/DATA_LAYER/split_data.py', '--config_path', 'script/config.yml'])