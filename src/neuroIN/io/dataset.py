from .utils import dir_file_types
from .edf import edf_to_np
from .gdf import gdf_to_np
from ..preprocessing.train_test_split import create_split
from ..preprocessing.crop import trim_length
from ..preprocessing.to_3d import dir_to_3d
from ..preprocessing.normalize import normalize_from_train

from pathlib import Path
import re
from math import sqrt
import time
import numpy as np
import pandas as pd
import json
import tqdm
import torch
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

neuroIN_extensions = {'.edf': edf_to_np, '.gdf': gdf_to_np}
neuroIN_data_path = Path('/Users/marktaylor/neuroIN/data')

def import_dataset(orig_dir, targ_dir, dataset_extensions, dataset_name=None, orig_f_pattern='*', id_regex=r'\d+', resample_freq=None, mapping=None):
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
    orig_path, targ_path = Path(orig_dir).expanduser(), Path(targ_dir).expanduser()

    assert orig_path.is_dir(), 'orig_dir must be a directory'
    targ_path.mkdir(parents=True, exist_ok=True)

    if not dataset_name:
        dataset_name = targ_path.stem
    annotations_f_name = targ_path / "annotations.csv"

    extensions = dir_file_types(orig_dir)
    assert extensions, "No files with supported extensions detected."
    ext_funcs = {ext: dataset_extensions[ext] for ext in extensions}

    dataset_params = {"name": dataset_name,
                      "orig_dir": str(orig_path.expanduser()),
                      "data_dir": str(targ_path.expanduser()),
                      "data_annotations_path": str(annotations_f_name.expanduser()),
                      "extensions": list(extensions)}

    trials_dict = {"np_file": [], "label": [], "subject_id": [],
                   "ext": [], "n_channels": [], "length": []}
    if not mapping: mapping = {}
    ch_names = []
    for ext, ext_func in ext_funcs.items():
        print(f"Processing {ext} files...")
        for orig_f in tqdm.tqdm(orig_path.rglob(orig_f_pattern + ext)):
            f_name = orig_f.stem
            subject_id = re.search(id_regex, f_name).group()

            try:
                if mapping:
                    trials, labels, mapping, ch_names = ext_func(orig_f, resample_freq=resample_freq, mapping=mapping)
                else:
                    trials, labels, mapping, ch_names = ext_func(orig_f, resample_freq=resample_freq)

                for i, trial in enumerate(trials):
                    npy_f = f'{f_name}_{i}.npy'
                    np.save(targ_path / npy_f, trial)

                    n_channels, length = trial.shape

                    for dict_name, item in zip(trials_dict.keys(), [npy_f, labels[i], subject_id, ext, n_channels, length]):
                        trials_dict[dict_name].append(item)
            except ValueError:
                print(f"{f_name} does not contain known labels, trials will be diregarded.")
        print(f"{ext} files processed.")

    pd.DataFrame(trials_dict).to_csv(annotations_f_name, index=False)

    # initialize more detailed dataset info for config file
    dataset_params["label_encoding"] = mapping
    dataset_params["n_dim"] = 2
    dataset_params["n_classes"] = len(mapping)
    dataset_params["split_into_sets"] = False
    dataset_params["preprocessing"] = []
    dataset_params["transform"] = None
    dataset_params["target_transform"] = None
    dataset_params["class_weights"] = None
    dataset_params["ch_names"] = ch_names
    dataset_params["model_optims"] = []

    # write config file in new dataset's directory
    with open(targ_path / "config.json", 'w') as stream:
        json.dump(dataset_params, stream, sort_keys=True, indent=4)

class Set(torchDataset):
    """Set is a subclass of torch.Dataset to hold sets of data for PyTorch

    Set is used for groups of data like training sets,
    while Dataset contains all data as well as metadata for a project.

    :param torchDataset: [description]
    :type torchDataset: [type]
    """

    def __init__(self, data_dir, transform=ToTensor(), target_transform=None):
        self.data_path = Path(data_dir).expanduser()
        assert self.data_path.is_dir(), "data_dir must be a directory"

        self.transform = transform
        self.target_transform = target_transform

        annotations_df = pd.read_csv(self.data_path / "annotations.csv")
        self.data_labels = annotations_df['label'].values
        self.np_files = annotations_df['np_file'].values
    
    def __len__(self):
        return len(self.data_labels)
    
    def __getitem__(self, idx):
        data = np.load((self.data_path / self.np_files[idx]))
        label = self.data_labels[idx]
        if self.transform:
                data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label
    
    def get_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    
    def get_annotation_df(self):
        return pd.read_csv(Path(self.data_path / "annotations.csv"))
    
    @property
    def df(self):
        return self.get_annotation_df()
    
    def mean_std(self, save=True):
        # initialize empty array with size channels (or 2D locations) by number of samples
        means = np.zeros((self[0][0].shape[:-1] + (len(self),)))

        # add means across time from each sample to array
        for i, (sample, _) in enumerate(self):
            means[..., i] = sample.mean(axis=-1)
        
        # find means and std for each channel (or 2D location) across samples
        mean = np.average(means, axis=-1)
        std = np.average((means - mean[..., None])**2, axis=-1)

        if save:
            np.save(self.data_path / 'mean.npy', mean)
            np.save(self.data_path / 'std.npy', mean)
        
        return mean, std
        
        
        

class Set3D(Set):
    """Set3D is a subclass of Set specifically for 3D data

    :param Set: [description]
    :type Set: [type]
    """
    def __getitem__(self, idx):
        """
        TODO: maybe call super() then add dimension??

        :param idx: [description]
        :type idx: [type]
        :return: [description]
        :rtype: [type]
        """
        data = np.load((self.data_path / self.np_files[idx]))
        data = torch.from_numpy(data).unsqueeze(0)
        label = self.data_labels[idx]

        if self.target_transform:
            label = self.target_transform(label)

        return data, label


class Dataset(Set):
    '''
    Dataset is a subclass of Set that contains all data for a project and associated metadata

    "config.json" contains parameters for the Dataset

    '''

    def __init__(self, data_dir, transform=ToTensor(), target_transform=None):
        super().__init__(data_dir, transform, target_transform)

        with open(self.data_path / "config.json", 'r') as stream:
            dataset_params = json.load(stream)

        for key, value in dataset_params.items():
            setattr(self, key, value)
    
    def update_param(self, key, value):
        with open(self.data_path / "config.json", 'r') as stream:
            dataset_params = json.load(stream)

        dataset_params[key] = value

        with open(self.data_path / "config.json", 'w') as stream:
            json.dump(dataset_params, stream, sort_keys=True, indent=4)
        
        setattr(self, key, value)


    # preprocessing methods
    def trim_data_length(self, length, start_i=0):
        trim_length(self.data_path, length, start_i=start_i)
        self.update_param("preprocessing", self.preprocessing + [f"trim_length: {length}"])

    def create_data_split(self, train_prop=0.7, val_prop=0.15, by_subject=False):
        create_split(self.data_path, train_prop=train_prop, val_prop=val_prop, by_subject=by_subject)
        self.update_param("split_into_sets", True)
        self.update_param("preprocessing", self.preprocessing + [f"create_split: train_prop={train_prop}, val_prop={val_prop}, by_subject={by_subject}"])
    
    def normalize_data_from_train(self):
        normalize_from_train(self.data_path)
        self.update_param("preprocessing", self.preprocessing + [f"normalized with train"])

    def data_to_3d(self, type_3d='d', d_idxs_override=None):
        dir_to_3d(self.data_path, type_3d=type_3d, ch_names=self.ch_names, d_idxs_override=d_idxs_override)
        self.update_param("n_dim", 3)
        self.update_param("preprocessing", self.preprocessing + [f"to_3d: type_3d={type_3d}"])
    
    def change_label_encoding(self, encoding_dict):
        for csv in self.data_path.rglob("annotations.csv"):
            annotations_df = pd.read_csv(csv)
            annotations_df['label'] = annotations_df['label'].apply(lambda x: encoding_dict[x])
            annotations_df.to_csv(csv, index=False)


    def get_set(self, set_name):
        assert self.split_into_sets == True, "Data has not been split into sets yet!"
        return Set(self.data_path / set_name, transform=self.transform, target_transform=self.target_transform)

    def get_sets(self):
        return [self.get_set(set_name) for set_name in ['train', 'val', 'test']]
    
    @property
    def train(self):
        return self.get_set('train')
    
    @property
    def val(self):
        return self.get_set('val')
    
    @property
    def test(self):
        return self.get_set('test')
    
    @property
    def sets(self):
        return self.get_sets()

    def get_dataloader(self, set_name, batch_size, shuffle=True):
        assert self.split_into_sets == True, "Data has not been split into sets yet!"
        return DataLoader(self.get_set(set_name), batch_size, shuffle=shuffle)

    def get_dataloaders(self, batch_size, shuffle=True):
        return [self.get_dataloader(set_name, batch_size, shuffle=shuffle) for set_name in ['train', 'val', 'test']]
    
    def standardize_channels(self, ch_names=None):
        if not ch_names: ch_names = [ch.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp') for ch in self.ch_names]
        self.update_param('ch_names', ch_names)
    
    def init_optim(self, config, fname=None):
        if not fname: fname = self.name + '_' + time.strftime("%Y%m%d-%H%M%S") + '.pt'

        self.optim_path = self.data_path / "optims"
        self.optim_path.mkdir(parents=True, exist_ok=True)

        f = str(self.optim_path / fname)
        torch.save(config, f)
        self.update_param("model_optims", self.model_optims + [f])

        print(f"Optimization configuration file saved to: {f}")
    
    def get_optim(self, i):
        return torch.load(self.model_optims[i])

    @property
    def last_optim(self):
        return torch.load(self.model_optims[-1])

class Dataset3D(Dataset):
    """Set3D is a subclass of Dataset specifically for 3D data

    :param Set: [description]
    :type Set: [type]
    """
    def __getitem__(self, idx):
        """
        TODO: maybe call super() then add dimension??

        :param idx: [description]
        :type idx: [type]
        :return: [description]
        :rtype: [type]
        """
        data = np.load((self.data_path / self.np_files[idx]))
        data = torch.from_numpy(data).unsqueeze(0)
        label = self.data_labels[idx]

        if self.target_transform:
            label = self.target_transform(label)

        return data, label