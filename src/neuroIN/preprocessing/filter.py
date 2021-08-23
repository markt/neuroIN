from neuroIN.io.dataset import Dataset

from pathlib import Path
import pandas as pd

def filter_labels(data_dir, labels_to_keep=[0, 1]):
    """Filter a Dataset to only contain trials with specified labels

    :param data_dir: the directory of the Dataset
    :type data_dir: str or pathlike
    :param labels_to_keep: the labels to keep, defaults to [0, 1]
    :type labels_to_keep: list, optional
    """
    data_path = Path(data_dir).expanduser()
    data = Dataset(data_path)

    new_df = data.df[data.df['label'].isin(labels_to_keep)]

    # delete all files 
    data.df[~data.df['label'].isin(labels_to_keep)]['np_file'].apply(lambda f: (data_path / f).unlink())

    new_df.to_csv((data_path / "annotations.csv"), index=False)

    data.update_param("n_classes", len(labels_to_keep))
    data.update_param("data_annotations_path", str(data_path / "annotations.csv"))