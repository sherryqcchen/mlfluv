import os
import yaml
import glob
import pandas as pd

def find_files_by_patern(folder, pattern):
    
    # Initialize an empty list to store matching files
    config_files = []

    # Use glob to find files matching the current pattern
    matching_files = glob.glob(os.path.join(folder, pattern))
    config_files.extend(matching_files)
    
    return config_files

def read_acc_from_log(log_path):
    # Initialize empty lists to store metrics
    mean_iou, micro_iou, macro_iou, accuracy, recall, precision, f1 = [], [], [], [], [], [], []

    # Open the log file and process each line
    inside_section = False
    with open(log_path, 'r') as file:
        lines = file.readlines()
        # print(file.read())
        for line in lines[-9:]:
            if "Overall" in line:
                inside_section = True
            elif inside_section:
                metric_name, metric_value = line.split("---->")
                metric_name = metric_name.split('-')[-1].strip()
                try:
                    metric_value = float(metric_value.strip())
                except ValueError:
                    values_str = metric_value.strip().strip('[]')  # Remove brackets
                    float_values = [float(val) for val in values_str.split()]
                if metric_name == "Mean IOU":
                    mean_iou.append(metric_value)
                elif metric_name == "Micro IOU":
                    micro_iou.append(metric_value)
                elif metric_name == "Macro IOU":
                    macro_iou.append(metric_value)
                elif metric_name == "Accuracy":
                    accuracy.append(metric_value)
                elif metric_name == "Recall":
                    recall.append(metric_value)
                elif metric_name == "Precision":
                    precision.append(metric_value)
                elif metric_name == "F1":
                    f1.append(metric_value)

    # Create a Pandas DataFrame
    df = pd.DataFrame({
        'Mean IoU': mean_iou,
        'Micro IoU': micro_iou,
        'Macro IoU': macro_iou,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1': f1
    })

    # Print the DataFrame
    # print(df)
    return df

def read_param_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    extracted_params = {}
    for section, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                extracted_params[key] = value
    
    df_params = pd.DataFrame([extracted_params])
    # print(df_params)
    return df_params

if __name__ == "__main__":
    print(os.getcwd())
    # Read the log file
    log_file = 'preds.log'
    config_file = 'config.yml'

    exp_root = 'script/experiments'

    exp_paths = sorted([os.path.join(exp_root, exp) for exp in os.listdir(exp_root)])
    print(exp_paths)
    dfs = []
    empty_finetue_acc = []
    for exp_path in exp_paths:

        init_config = find_files_by_patern(exp_path, 'config*.yml')[0]
        init_accuracy = find_files_by_patern(exp_path, 'preds.log')[0]
        print(init_accuracy)

        df_acc_init = read_acc_from_log(init_accuracy)
        df_param_init = read_param_from_config(init_config)

        # Check if accuracy df is empty
        if df_acc_init.empty:
            empty_finetue_acc.append(exp_path)
            print(f"The overall accuracy of exp. {exp_path} is not calculated.")
        else:
            # Join these two dataframes
            merged_df_init = pd.concat([df_param_init, df_acc_init], axis=0, ignore_index=True)
            # Stack two rows to one row
            df_ft_init = pd.DataFrame([merged_df_init.stack().values], columns=merged_df_init.columns)
            dfs.append(df_ft_init)

        fine_tune_paths = sorted([os.path.join(exp_path, folder) for folder in os.listdir(exp_path) if folder.startswith('fine_tune')])

        # print(len(fine_tune_paths))
        # print(fine_tune_paths[0])
        
        for fine_tune_path in fine_tune_paths:
            fine_tune_config = find_files_by_patern(fine_tune_path, 'config*.yml')[0]
            fine_tune_accuracy = find_files_by_patern(fine_tune_path, 'preds.log')[0]
            
            df_acc = read_acc_from_log(fine_tune_accuracy)

            # Check if accuracy df is empty
            if df_acc.empty:
                empty_finetue_acc.append(fine_tune_path)
                print(f"The overall accuracy of exp. {fine_tune_path} is not calculated.")
                continue

            df_param = read_param_from_config(fine_tune_config)

            # Join these two dataframes
            merged_df = pd.concat([df_param, df_acc], axis=0, ignore_index=True)
            # Stack two rows to one row
            df_ft = pd.DataFrame([merged_df.stack().values], columns=merged_df.columns)
            dfs.append(df_ft)

    df = pd.concat(dfs, ignore_index=True)
    print(df)

    with open('empty_dfs.txt', 'w') as file:
        for df_name in empty_finetue_acc:
            file.write(df_name + '\n')
