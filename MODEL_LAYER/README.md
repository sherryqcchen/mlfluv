# MLFluvUnet: a Fluvial U-Net Segmentation Model

MLFluvUnet is a U-Net based model for multi-class segmentation of remote sensing images. It is built on top of PyTorch and designed specifically for fluvial environments.

## Directory Structure

The main scripts are located in the `MODEL_LAYER` directory:

- `dataset.py`: Defines the `MLFluvDataset()` class, a PyTorch dataset class that ingests data created by the scripts in the `DATA_LAYER` folder.

- `model.py`: Defines the `SMPUnet()` class, which wraps a basic U-Net model from the  [Segmentation Models.Pytorch](https://github.com/qubvel/segmentation_models.pytorch) package. It includes additional functionalities such as masking data, freezing/defreezing encoders, and loading pre-trained models with a modified number of classes.

- `interface.py`: Defines the `MLFluvUnetInterface()` class, which includes functions for training on one epoch, validating the model, and training on many epochs.

- `train.py`: Used for training an initial U-Net model from a general land use and land cover (LULC) map. In this study, we choose ESRI land cover map. 

- `inference.py`: Used for making predictions with a model that is trained by `train.py`.

- `fine_tune.py`: Used for incremental learning on the initial U-Net model. The trained model will be loaded and modified for incremental classes (e.g., the fluvial sediment class) in the new dataset.

## Getting Started

To train the initial U-Net model, run:

```bash
python train.py
```

To make predictions with the trained model, run:
```bash
python inference.py
```

To perform incremental learning on the initial U-Net model, run:
```bash
python fine_tune.py
```

