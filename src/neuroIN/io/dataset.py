from .utils import dir_file_types
from .edf import edf_to_np

from pathlib import Path
import re
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

neuroIN_extensions = {'.edf': edf_to_np}
neuroIN_data_path = Path('/Users/marktaylor/neuroIN/data')

def import_dataset(orig_dir, targ_dir, dataset_name=None, orig_f_pattern='*', id_regex=r'\d+'):
    """Import a new dataset into neuroIN.

    This function imports a dataset of files with recognized extensions into
    a target directory of NumPy files; annotation information (such as labels)
    is written to the same directory. A configuration file is initialized
    in the datasets target directory and information on the dataset is appended
    to a configuration file containing information on all imported datasets.

    :param orig_dir: The directory to import data from
    :type orig_dir: string or pathlike
    :param targ_dir: The directory to import data to
    :type targ_dir: string or pathlike
    :param dataset_name: The name of the dataset, defaults to None which uses the targ_dir name
    :type dataset_name: string, optional
    :param orig_f_pattern: The pattern of files to import, defaults to '*'
    :type orig_f_pattern: str, optional
    :param id_regex: The RegEx to parse subject ID's from files, defaults to r'\d+'
    :type id_regex: regexp, optional
    """
    orig_path, targ_path = Path(orig_dir), Path(targ_dir)

    assert orig_path.is_dir(), 'orig_dir must be a directory'
    targ_path.mkdir(parents=True, exist_ok=True)

    if not dataset_name:
        dataset_name = targ_path.stem
    annotations_f_name = targ_path / "annotations.csv"

    extensions = dir_file_types(orig_dir)
    assert extensions, "No files with supported extensions detected."
    ext_funcs = {ext: neuroIN_extensions[ext] for ext in extensions}

    dataset_params = {"name": dataset_name,
                      "orig_dir": str(orig_path.expanduser()),
                      "data_dir": str(targ_path.expanduser()),
                      "data_annotations": str(annotations_f_name.expanduser()),
                      "extensions": list(extensions)}

    trials_dict = {"np_file": [], "label": [], "subject_id": [],
                   "ext": [], "n_channels": [], "length": []}
    mapping = {}
    for ext, ext_func in ext_funcs.items():
        for orig_f in orig_path.rglob(orig_f_pattern + ext):
            f_name = orig_f.stem
            subject_id = re.search(id_regex, f_name).group()

            trials, labels, file_mapping = ext_func(orig_f)
            mapping.update(file_mapping)

            for i, trial in enumerate(trials):
                npy_f = f'{f_name}_{i}.npy'
                np.save(targ_path / npy_f, trial)

                n_channels, length = trial.shape

                for dict_name, item in zip(trials_dict.keys(), [npy_f, labels[i], subject_id, ext, n_channels, length]):
                    trials_dict[dict_name].append(item)

    pd.DataFrame(trials_dict).to_csv(annotations_f_name, index=False)

    # initialize more detailed dataset info for config file
    dataset_params["mapping"] = mapping
    dataset_params["n_dim"] = 2
    dataset_params["transform"] = None
    dataset_params["target_transform"] = None
    dataset_params["weights"] = None

    # write config file in new dataset's directory
    with open(targ_path / "dataset_info.json", 'w') as stream:
        json.dump(dataset_params, stream, sort_keys=True, indent=4)

class Dataset(torchDataset):
    '''
    Subclass of PyTorch Dataset to store data for PyTorch
    '''

    def __init__(self, data_dir, transform=ToTensor(), target_transform=None):
        self.data_path = Path(data_dir).expanduser()
        assert self.data_path.is_dir(), "data_dir must be a directory"

        with open(self.data_path / "dataset_info.json", 'r') as stream:
            dataset_params = json.load(stream)

        print(dataset_params)
        for key, value in dataset_params.items():
            setattr(self, key, value)
        
        if not self.transform: self.transform = transform
        if not self.target_transform: self.target_transform = target_transform
        if not self.data_annotations: self.data_annotations = self.data_path / "annotations.csv"

        self.annotations_df = pd.read_csv(Path(self.data_annotations))
        self.data_labels = self.annotations_df['label'].values
        self.np_files = self.annotations_df['np_file'].values

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        data = np.load((self.data_path / self.np_files[idx]))
        label = self.data_labels[idx]
        if data.ndim > 2:
            data = torch.from_numpy(data).unsqueeze(0)
        else:
            if self.transform:
                data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label
    
    def get_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)