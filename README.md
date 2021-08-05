# neuroIN (neuroImageNets)

neuroIN is a deep learning package for neuroimaging data.

The basic workflow for neuroIN is:
-process EEG (other modalities to come) data into image and video like arrays
-initialize an architecture for your data
-train network and optimize hyperparameters
-visualize results and network features


## Overview of modules

### io

`neuroin.io` contains functions and classes associated with importing datasets from neuroimaging data formats into NumPy arrays. Information on imported datasets is stored in configuration files that are used to load `Dataset` objects so imported datasets can be used.

### preprocess

`neuroin.preprocess` contains functions for preprocessing imported data; this entails training/testing split generation, data normalization, and data augmentation methods.

### models

`neuroin.models` contains classes for the different architectures supported by the package.

### training

`neuroin.training` contains function to train models, classify data, log training history, and save trained models.

### optim

`neuroin.optim` contains functions for optimizing network hyperparameters.

### vis

`neuroin.vis` contains visualization functions for training results and feature visualization.