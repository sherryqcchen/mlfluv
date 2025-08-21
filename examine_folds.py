import os
import numpy as np

data_path = "data/fold_data/finetune_with_urban_bare_DW_5_fold"

# Load all files from the data path
file_paths = [os.path.join(data_path, file) for file in os.listdir(data_path)]
print(len(file_paths))

all_folds = [np.load(file, allow_pickle=True) for file in file_paths if file.endswith('.npy')]
print(len(all_folds[0]))

# Identify which folds are inconsistent
folds = [0, 1, 2, 3, 4]
for index in folds:
    if all_folds[index].ndim == 1:
        print('Find the weird fold')
        print(type(all_folds[index]))
        print(len(all_folds[index]))
        for list in all_folds[index]:
            if len(list) != 3:
                print(list)
    else:
        print(f"Fold {index} is already an ndarray with shape: {all_folds[index].shape}")

# Concatenate the folds
try:
    data = np.concatenate([all_folds[idx] for idx in folds], axis=0)
except ValueError as e:
    print("Error during concatenation:", e)
    for index in folds:
        print(f"Shape of fold {index}: {all_folds[index].shape}")
        print(f"Type of elements in fold {index}: {type(all_folds[index][0][0])}")
