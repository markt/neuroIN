from neuroIN.io.dataset import Dataset
from neuroIN.models.multi import MultiBranchCNN2D

import os
from pathlib import Path
from ray import tune

data = Dataset("~/neuroIN/data/eegmmidb_multi")

config = {
    "data_dir": data.data_path, 
    "checkpoint_dir": data.data_path / "checkpoints", 
    "model": MultiBranchCNN2D,
    "dropout_p": tune.uniform(0, 0.8),
    "lr": tune.loguniform(1e-4, 1e-1),
    "momentum": tune.uniform(0.1, 0.9),
    "batch_size": tune.choice([8, 16]),
    # "tk_base": tune.randint(100, 140),
    # "tk_factor": tune.uniform(1.5, 2),
    # "temp_kernel_lengths": tune.sample_from(lambda spec: [spec.config.tk_base/spec.config.tk_factor,
    #                                                       spec.config.tk_base,
    #                                                       spec.config.tk_base*spec.config.tk_factor]),
    # "temp_kernel_lengths": [int(tune.randint(110, 130)*(tune.uniform(1.5, 2)**float(e))) for e in [-1, 0, 1]],
    "temp_kernel_lengths": [tune.randint(60, 100),
                            tune.randint(110, 140),
                            tune.randint(180, 220)],
    "spac_kernel_lengths": [tune.randint(8, 22)]*3,
    "F1": tune.randint(4, 8),
    "D": tune.choice([1, 2]),
    "F2": tune.randint(8, 12),
    "fc_size": tune.choice([16, 32])
}

data.init_optim(config)