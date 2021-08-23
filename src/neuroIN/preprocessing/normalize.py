from neuroIN.io import dataset

from pathlib import Path
import numpy as np

def normalize_from_train(data_dir):
    """Normalize the Dataset using the mean and std dev of the training Set (requires data to be split into sets)

    :param data_dir: the directory of the Dataset
    :type data_dir: str or pathlike
    """
    data_path = Path(data_dir)
    data = dataset.Dataset(data_path)

    train_set = data.train
    train_mean, train_std = train_set.mean_std()


    for f in data_path.rglob("*.npy"):
        ar = np.load(f)
        ar = (ar - train_mean[..., None]) / train_std[..., None]
        np.save(f, np.float32(ar))