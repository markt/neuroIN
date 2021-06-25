import os

import numpy as np
import pandas as pd

from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from mne import events_from_annotations
from mne.channels.montage import DigMontage
from mne.datasets import eegbci # CONSIDER IF SHOULD BE .EDF SOURCE AGNOSTIC

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


def load_raw_edf(fname, ch_names=None, montage_kind='standard_1005', verbose=False):
    '''
    Loads a trial by filename and returns an MNE RawEDF object

    ch_names is an optional dictionary for renaming channels
    montage_kind is an optional DigMontage or montage_kind string
    '''
    raw = read_raw_edf(fname, preload=True, verbose=verbose)

    # check if new channel names specified, otherwise create them
    if ch_names:
        assert isinstance(ch_names, dict), 'ch_names must be a dictionary'
    else:
        new_names = {ch: ch.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp') for ch in raw.ch_names}
    raw.rename_channels(new_names)

    # check if montage object specified, otherwise assume montage is string and create one
    if isinstance(montage_kind, str):
        montage = make_standard_montage(montage_kind)
    else:
        assert isinstance(montage_kind, DigMontage), 'montage_kind must be a string or DigMontage object'
        montage = montage_kind
    raw.set_montage(montage)

    return raw

def edf_to_numpy(edf, max_length=None, save=False, data_dir='data', prefix='event'):
    '''
    Converts an MNE RawEDF object into a list of numpy arrays for each event and list of corresponding labels

    Can optionally save each numpy array into a .csv with an associated file 'annotations.csv' containing labels
    '''
    data, times = edf[:, :]
    events, mapping = events_from_annotations(edf)

    # reverse mapping gives annotations from data
    rmapping = {v: k for k, v in mapping.items()}    

    trials = []
    labels = []

    for i in range(len(events) - 1):
        start, end, label = events[i][0], events[i+1][0], rmapping[events[i][2]]
        trials.append(data[:,start:end])
        labels.append(label)
    # handle last event separately as there is no next index
    start, _, label = events[-1][0], data.shape[1], rmapping[events[-1][2]]
    trials.append(data[:,start:])
    labels.append(label)

    if max_length: # could this maybe be a transform??
        trials = [t[:,:max_length] for t in trials]

    if save:
        if not os.path.exists(f'{os.getcwd()}/{data_dir}'):
            os.mkdir(f'{os.getcwd()}/{data_dir}')
        
        with open(f'{data_dir}_annotations.csv', 'a') as f:
            for i in range(len(trials)):
                fname = f'{prefix}_{i}.csv'
                np.savetxt(f'{data_dir}/{fname}', trials[i], delimiter=',')
                f.write(f'{fname}, {labels[i]}\n')

    return trials, labels


def save_subjects(subject_dict, runs, max_length=None, verbose=False):
    '''
    Saves data from .EDF files into .csv files for PyTorch
    
    subject_dict stores each group of subjects (e.g. training) with a corresponding list of ID's
    '''
    dataset_dict = {}
    
    for group in subject_dict:
        open(f'{group}_annotations.csv', 'w').close()
        
        for subject in subject_dict[group]:
            for fname in eegbci.load_data(subject, runs, verbose=verbose):
                edf = load_raw_edf(fname, verbose=verbose)
                
                pre_name = fname.split('/')[-1].split('.')[0]
                trials, labels = edf_to_numpy(edf, max_length=max_length, save=True, data_dir=group, prefix=pre_name)


class eegDataset(Dataset):
    '''
    Subclass of PyTorch Dataset to prepare EEG files for PyTorch

    Each EEG event is stored in a .csv file contained in data_dir
    annotations_file stores the name of each .csv and corresponding label
    '''

    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        self.data_labels = pd.read_csv(annotations_file, header=None)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data_labels)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_labels.iloc[idx, 0])
        data = pd.read_csv(data_path, header=None).to_numpy()
        label = self.data_labels.iloc[idx, 1]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label


def load_subjects(groups, parent_dir=os.getcwd(), transform=ToTensor(), batch_size=1, shuffle=True):
    '''
    Loads groups of data into PyTorch DataLoader objects
    '''
    data_loaders = []
    
    if isinstance(groups, dict):
        groups = [*groups]
    
    for group in groups:
        dataset = eegDataset(f'{group}_annotations.csv', f'{parent_dir}/{group}', transform=ToTensor())
        data_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
    
    return data_loaders