from neuroIN.io.dataset import Dataset
from pathlib import Path
import numpy as np
import pandas as pd

def merge_datasets(data_dir1, data_dir2, ch_names, targ_dir):
    """Merge two datasets into one.

    Only channels included in both Datasets will be used.

    :param data_dir1: the directory of the first Dataset
    :type data_dir1: str or pathlike
    :param data_dir2: the directory of the second Dataset
    :type data_dir2: str or pathlike
    :param ch_names: channel names to include for the two Datasets, does not need to be same order for both Datasets
    :type ch_names: list
    :param targ_dir: the directory for the merged Dataset
    :type targ_dir: str or pathlike
    """
    targ_path = Path(targ_dir)
    targ_path.mkdir(parents=True, exist_ok=True)

    dataset1, dataset2 = Dataset(data_dir1), Dataset(data_dir2)

    labels = set(dataset1.df['label'].unique()).intersection(set(dataset2.df['label'].unique()))

    joined_df = pd.concat([dataset1.df, dataset2.df])

    joined_df = joined_df[joined_df['label'].isin(labels)]
    joined_df.to_csv(targ_path / 'annotations.csv')

    for data_dir in [data_dir1, data_dir2]:
        dataset = Dataset(data_dir)

        assert set(ch_names).issubset(dataset.ch_names)

        dataset_ch_names_to_idx = {name: dataset.ch_names.index(name) for name in dataset.ch_names}
        dataset_idxs = [dataset_ch_names_to_idx[n] for n in ch_names]

        for f in Path(data_dir).rglob('*.npy'):
            if joined_df['np_file'].str.contains(f.name).any():
                data = np.load(f)
                data = data[dataset_idxs,:]

                np.save(targ_path / f.name, data)