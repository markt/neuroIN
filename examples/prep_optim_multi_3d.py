from neuroIN.io.dataset import Dataset
from neuroIN.models.multi import MultiBranchCNN3D

import os
from pathlib import Path
from ray import tune

data = Dataset("~/neuroIN/data/eegmmidb_multi_3d")

config = {
    "data_dir": data.data_path, 
    "checkpoint_dir": data.data_path / "checkpoints", 
    "model": MultiBranchCNN3D,
    "dropout_p": tune.uniform(0, 0.8),
    "lr": tune.loguniform(1e-4, 1e-1),
    "momentum": tune.uniform(0.1, 0.9),
    "batch_size": tune.choice([4, 8, 16, 32]),
    "fc_size": tune.choice([16, 32])
}

data.init_optim(config)