# Identifying Fluvial Sediment Using Remote Sensing Data and Deep Learning

This repository contains the code and scripts for the manuscript **Identifying fluvial sediment using remote sensing data and deep learning**. It implements a streamlined process to prepare datasets, configure training pipelines, and conduct hyperparameter optimization for incremental learning using deep learning models (e.g., U-Net).

---

## Purpose

The purpose of this repository is to showcase the process of adding a new land cover class to existing land cover maps and refining the representation of such a class. The example new class demonstrated in the manuscript and scripts is **fluvial sediment**. This new class is learned using incremental learning techniques, providing an efficient and scalable approach to enhance existing land cover classifications.

The incremental learning module is based on the study by Michieli and Zanuttigh (2021) ["Knowledge Distillation for Incremental Learning in Semantic Segmentation"](https://arxiv.org/abs/1911.03462). In this repository, the incremental learning implementation has been re-written using the PyTorch framework, as opposed to the TensorFlow implementation in the original study.

## Repository Structure

The repository is divided into three main layers:

1. **DATA_LAYER**
   - Contains scripts and documentation for preparing the dataset used in this study.
   - Includes functionality to streamline preprocessing steps, generate training-ready datasets, and manage dataset configurations.

2. **MODEL_LAYER**
   - Houses scripts and documentation related to the U-Net model, including its architecture and configurations.
   - Provides tools for model training, fine-tuning, and incremental learning.

3. **ANALYSE_LAYER**
   - Includes scripts for reproducing the analysis and generating results as shown in the manuscript.
   - Focuses on evaluating model performance and visualizing results.

---

## Key Scripts and Configurations

### `config.yml`
This YAML file is the central configuration hub for all training and model parameters. Users can modify this file to:
- Adjust dataset paths and settings.
- Set hyperparameters for model training, including learning rates, batch sizes, and epochs.
- Configure options for incremental learning steps, such as number of classes, class weights and temperature scaling.

### `prepare_mlfluv_dataset.py`
This script automates the dataset preparation process, ensuring consistent and reproducible workflows. Key features include:
- Reading raw data from specified directories.
- Preprocessing images (e.g., normalization, resizing) and labels.
- Generating training, validation, and test splits.
- Saving the processed dataset in formats compatible with training pipelines.

To run the script:
```bash
python prepare_mlfluv_dataset.py
```

### `train_model.py`
This script handles the model training process, focusing on hyperparameter tuning for incremental learning using grid search. Key features include:
- Training the U-Net model with different combinations of hyperparameters.
- Logging training results and metrics for each hyperparameter combination.
- Supporting incremental training setups to refine segmentation accuracy over successive iterations.

To run the script:
```bash
python train_model.py
```


## How to Reproduce the Analysis
1. Clone this repository:
   ```bash
   git clone https://github.com/sherryqcchen/mlfluv.git
   cd mlfluv
   ```
2. Install the required dependencies (check `requirements.txt`).
3. Prepare the dataset:
   ```bash
   python prepare_mlfluv_dataset.py
   ```
4. Train the model:
   ```bash
   python train_model.py
   ```
5. Analyze and visualize results using scripts in the `ANALYSE_LAYER`.

---

## License

Copyright [2024] [Qiuyang Chen]

Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at:

```
   http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

---

Feel free to raise an issue or contact the repository owner for further questions or clarifications!
