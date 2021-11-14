from .utils import dir_file_types
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

def import_dataset(orig_dir, targ_dir, dataset_extensions, dataset_name=None, orig_f_pattern='*', id_regex=r'\d+', resample_freq=None, mapping=None,  np_dtype=np.float32):
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
    :param dataset_extensions: A dictionary mapping file extensions to functions processing those files into NumPy arrays
    :type dataset_extensions: dict
    :param dataset_name: A name for the dataset, defaults to None
    :type dataset_name: str, optional
    :param orig_f_pattern: The pattern used to search for files to import, defaults to '*'
    :type orig_f_pattern: str, optional
    :param id_regex: A RegEx for extracting subject ID from a filename, defaults to r'\d+'
    :type id_regex: regexp, optional
    :param resample_freq: A new frequency to resample the data to, defaults to None
    :type resample_freq: int, optional
    :param mapping: a dictionary mapping labels in the data to labels to be used for classification, defaults to None
    :type mapping: dict, optional
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
                    trials, labels, mapping, ch_names = ext_func(orig_f, resample_freq=resample_freq, np_dtype=np_dtype, mapping=mapping)
                else:
                    trials, labels, mapping, ch_names = ext_func(orig_f, resample_freq=resample_freq, np_dtype=np_dtype)

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
        """Return a dataloader for the Set

        :param batch_size: The batch size for the DataLoader
        :type batch_size: int
        :param shuffle: Whether to shuffle the data or not, defaults to True
        :type shuffle: bool, optional
        :return: A DataLoader for the set
        :rtype: torch.utils.data.DataLoader
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    
    def get_annotation_df(self):
        """Returns the annotation DataFrame for the Set

        :return: The annotation DataFrame
        :rtype: pandas.DataFrame
        """
        return pd.read_csv(Path(self.data_path / "annotations.csv"))
    
    @property
    def df(self):
        return self.get_annotation_df()
    
    def mean_std(self, save=True):
        """Return the mean and standard deviation of the Set

        :param save: whether to save the mean and std dev arrays, defaults to True
        :type save: bool, optional
        :return: A mean array and a std dev array
        :rtype: numpy.array
        """
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


class Dataset(Set):
    '''
    Dataset is a subclass of Set that contains all data for a project and associated metadata

    "config.json" contains parameters for the Dataset that will be added as attributes
    '''

    def __init__(self, data_dir, transform=ToTensor(), target_transform=None):
        super().__init__(data_dir, transform, target_transform)

        with open(self.data_path / "config.json", 'r') as stream:
            dataset_params = json.load(stream)

        for key, value in dataset_params.items():
            setattr(self, key, value)
    
    def update_param(self, key, value):
        """Update a parameter for the dataset, update is reflect in config.json file as well

        :param key: the name of the parameter to update
        :type key: str
        :param value: the value to update the parameter with
        :type value: any JSON serializable
        """
        with open(self.data_path / "config.json", 'r') as stream:
            dataset_params = json.load(stream)

        dataset_params[key] = value

        with open(self.data_path / "config.json", 'w') as stream:
            json.dump(dataset_params, stream, sort_keys=True, indent=4)
        
        setattr(self, key, value)


    # preprocessing methods
    def trim_data_length(self, length, start_i=0):
        """Trims the length of all trials in a dataset, discarding those that are too short

        :param length: the length to trim to
        :type length: int
        :param start_i: the index to start trimming from, defaults to 0
        :type start_i: int, optional
        """
        trim_length(self.data_path, length, start_i=start_i)
        self.update_param("preprocessing", self.preprocessing + [f"trim_length: {length}"])

    def create_data_split(self, train_prop=0.8, val_prop=0, by_subject=False):
        """Split the Dataset into training, validation and testing sets

        :param train_prop: the proportion of data to go in the training set, defaults to 0.8
        :type train_prop: float, optional
        :param val_prop: the proportion of data to go in the validation set, defaults to 0
        :type val_prop: float, optional
        :param by_subject: whether to create splits by subject or by trials, defaults to False
        :type by_subject: bool, optional
        """
        create_split(self.data_path, train_prop=train_prop, val_prop=val_prop, by_subject=by_subject)
        self.update_param("split_into_sets", True)
        self.update_param("preprocessing", self.preprocessing + [f"create_split: train_prop={train_prop}, val_prop={val_prop}, by_subject={by_subject}"])
    
    def normalize_data_from_train(self):
        """Normalize the Dataset using the mean and std dev of the training Set (requires data to be split into sets)
        """
        normalize_from_train(self.data_path)
        self.update_param("preprocessing", self.preprocessing + [f"normalized with train"])

    def data_to_3d(self, type_3d='d', d_idxs_override=None):
        """Convert all trials in the Dataset into 3D format

        :param type_3d: the type of 3D configuration to use, defaults to 'd'
        :type type_3d: str, optional
        :param d_idxs_override: indices for D configuration that can be used to override the automatic index selection, defaults to None
        :type d_idxs_override: list, optional
        """
        dir_to_3d(self.data_path, type_3d=type_3d, ch_names=self.ch_names, d_idxs_override=d_idxs_override)
        self.update_param("n_dim", 3)
        self.update_param("preprocessing", self.preprocessing + [f"to_3d: type_3d={type_3d}"])
    
    def change_label_encoding(self, encoding_dict):
        """Change the encoding of labels to something else (note that PyTorch requires labels to be integers monotonically increasing starting from 0)

        :param encoding_dict: The encoding to use for the labels
        :type encoding_dict: dict
        """
        for csv in self.data_path.rglob("annotations.csv"):
            annotations_df = pd.read_csv(csv)
            annotations_df['label'] = annotations_df['label'].apply(lambda x: encoding_dict[x])
            annotations_df.to_csv(csv, index=False)


    def get_set(self, set_name):
        """Get a Set by name (nearly always should be either 'train', 'val', or 'test')

        :param set_name: the name of the Set to get
        :type set_name: str
        :return: the requested Set
        :rtype: Set
        """
        assert self.split_into_sets == True, "Data has not been split into sets yet!"
        return Set(self.data_path / set_name, transform=self.transform, target_transform=self.target_transform)

    def get_sets(self):
        """Get the training, validation, and test sets

        :return: The list of Sets
        :rtype: list
        """
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
        """Get a DataLoader for a Set by name (nearly always should be either 'train', 'val', or 'test')

        :param set_name: the name of the Set to get
        :type set_name: str
        :param batch_size: the batch size to use
        :type batch_size: int
        :param shuffle: whether or not to shuffle the data, defaults to True
        :type shuffle: bool, optional
        :return: A DataLoader for the set
        :rtype: torch.utils.data.DataLoader
        """
        assert self.split_into_sets == True, "Data has not been split into sets yet!"
        return DataLoader(self.get_set(set_name), batch_size, shuffle=shuffle)

    def get_dataloaders(self, batch_size, shuffle=True):
        """Get DataLoaders for the training, validation, and testing Sets.

        :param batch_size: the batch size to use
        :type batch_size: int
        :param shuffle: whether or not to shuffle the data, defaults to True
        :type shuffle: bool, optional
        :return: the list of DataLoaders
        :rtype: list
        """
        return [self.get_dataloader(set_name, batch_size, shuffle=shuffle) for set_name in ['train', 'val', 'test']]
    
    def standardize_channels(self, ch_names=None):
        """Standardize the Dataset's channel names

        :param ch_names: channel names to replace with, defaults to None
        :type ch_names: list, optional
        """
        if not ch_names: ch_names = [ch.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp') for ch in self.ch_names]
        self.update_param('ch_names', ch_names)
    
    def init_optim(self, config, fname=None):
        """Initialize parameters for an optimization experiment

        :param config: the parameter configuration dictionary to use
        :type config: dict
        :param fname: the file name to use for the configuration data, defaults to None
        :type fname: str, optional
        """
        if not fname: fname = self.name + '_' + time.strftime("%Y%m%d-%H%M%S") + '.pt'

        self.optim_path = self.data_path / "optims"
        self.optim_path.mkdir(parents=True, exist_ok=True)

        f = str(self.optim_path / fname)
        torch.save(config, f)
        self.update_param("model_optims", self.model_optims + [f])

        print(f"Optimization configuration file saved to: {f}")
    
    def get_optim(self, i):
        """Get the i'th optimization configuration dictionary

        :param i: the index of the dictionary to get
        :type i: int
        :return: the optimization configuration dictionary
        :rtype: dict
        """
        return torch.load(self.model_optims[i])

    @property
    def last_optim(self):
        return torch.load(self.model_optims[-1])