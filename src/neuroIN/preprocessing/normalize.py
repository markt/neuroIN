from neuroIN.io import dataset

from pathlib import Path
import numpy as np

def normalize_from_train(data_dir):
    data_path = Path(data_dir)
    data = dataset.Dataset(data_path)

    train_set = data.train
    train_mean, train_std, _ = train_set.mean_std()


    for f in data_path.rglob("*.npy"):
        ar = np.load(f)
        print(f"{f}: {ar.shape}")
        ar = (ar - train_mean[..., None]) / train_std[..., None]
        np.save(f, np.float32(ar))