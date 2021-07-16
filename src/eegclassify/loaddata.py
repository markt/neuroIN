import os
from pathlib import Path
import shutil
import re

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

def edf_to_numpy(edf, mapping='auto', length=None, np_dtype=np.float32):
    '''
    Converts an MNE RawEDF object into a list of numpy arrays for each event and list of corresponding labels

    Can optionally save each numpy array into a .csv with an associated file 'annotations.csv' containing labels
    
    Can take in and return label_mapping
    (LABEL_MAPPING INPUT MAY BE REDUNDANT IF CONSISTENT ACROSS ALL TRIALS?? ONLY HAVE TO ENSURE LABELING IS
    CONSISTENT ACROSS SUBJECTS, CONSIDER REMOVING)
    ADD UNIT TEST PERHAPS?
    '''
    data, times = edf[:, :]
    data = np_dtype(data)
    events, mapping = events_from_annotations(edf, mapping)

    # if not mapping:
    #     mapping = default_mapping
    # reverse mapping gives annotations from data
    # rmapping = {v: k for k, v in mapping.items()}    

    trials = []
    labels = []

    for i in range(len(events) - 1):
        start, end, label = events[i][0], events[i+1][0], events[i][2]
        trials.append(data[:,start:end])
        labels.append(label)
    # handle last event separately as there is no next index
    start, _, label = events[-1][0], data.shape[1], events[-1][2]
    trials.append(data[:,start:])
    labels.append(label)

    if length: # could this maybe be a transform??
        trials = [t[:,:length] for t in trials]

    return trials, labels


def save_subjects(subject_dict, runs, length=None, mapping='auto', verbose=False):
    '''
    Saves data from .EDF files into .csv files for PyTorch
    
    subject_dict stores each group of subjects (e.g. training) with a corresponding list of ID's
    '''
    for group in subject_dict:
        open(f'{group}_annotations.csv', 'w').close() # clears file
        
        for subject in subject_dict[group]:
            for fname in eegbci.load_data(subject, runs, verbose=verbose):
                edf = load_raw_edf(fname, verbose=verbose)
                
                pre_name = fname.split('/')[-1].split('.')[0]
                trials, labels = edf_to_numpy(edf, mapping=mapping, length=length)


class eegDataset(Dataset):
    '''
    Subclass of PyTorch Dataset to prepare EEG files for PyTorch

    Each EEG event is stored in a .csv file contained in data_dir
    annotations_file stores the name of each .csv and corresponding label
    '''

    def __init__(self, data_dir, transform=ToTensor(), target_transform=None):
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(self.data_dir / 'annotations.csv')
        self.data_labels = self.df['label'].values
        self.csv_files = self.df['csv_file'].values
        self.transform = transform
        self.target_transform = target_transform

        # mapping = pd.read_csv(mapping_file, header=None, index_col=False).to_dict('split')
        # self.mapping = {d[0]: d[1] for d in mapping['data']}
    
    def __len__(self):
        return len(self.data_labels)
    
    def __getitem__(self, idx):
        # data_path = os.path.join(self.data_dir, self.data_labels.iloc[idx, 0])
        # data = pd.read_csv((self.data_dir / self.csv_files[idx]), header=None).to_numpy()
        data = np.load((self.data_dir / self.csv_files[idx]))
        label = self.data_labels[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label


def load_subjects(groups, parent_dir=Path.cwd(), transform=ToTensor(), batch_size=1, shuffle=True):
    '''
    Loads groups of data into PyTorch DataLoader objects
    '''
    parent_dir = Path(parent_dir)
    assert parent_dir.is_dir(), 'parent_dir must be a directory'

    annotation_path = parent_dir / 'annotations.csv'
    if not annotation_path.is_file():
        print(f'No annotation file found, saving data from .EDF files into .CSVs...')
        save_subjects


    data_loaders = []
    
    for group in groups:
        group_dir = parent_dir / group

        if not group_dir.is_dir():
            print(f'No directory exists for {group} group')

        dataset = eegDataset(f'{parent_dir}/{group}_annotations.csv', f'{parent_dir}/{group}', transform=ToTensor())
        data_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
    
    return data_loaders

def load_eegmmidb(data_dir, between_subjects=True, edf_dir=None, mapping={'T0': 0, 'T1': 1, 'T2': 2}, length=640, verbose=True):
    '''
    data_dir: path that .CSVs are located or will be saved to
    between_subects: boolean for between or within subjects if training/testing split not present
    edf_dir: directory where .EDFs are located
    mapping: mapping of original labels to PyTorch labels
    length: length of a given trial (sampling rate * seconds)
    verbose: verbose output
    '''

    data_path = Path(data_dir)
    assert data_path.is_dir(), 'parent_dir must be a directory'

    annotation_path = data_path / 'annotations.csv'
    if not annotation_path.is_file():
        print('annotations.csv file not found, generating data from .EDFs...')

        if not edf_dir:
            try:
                edf_dir = os.environ['MNE_DATASETS_EEGBCI_PATH']
                edf_path = Path(edf_dir)
            except KeyError:
                # MNE_DATASETS_EEGBCI_PATH not set, using default location
                edf_path = Path().home() / 'mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/'
        else:
            edf_path = Path(edf_dir)

        annotation_df_rows = []
        for edf_f in edf_path.rglob('*.edf'):
            subject_id_run = str(edf_f).split('/')[-1].split('.')[0]
            subject_id = re.search(r'\d+', subject_id_run).group()

            edf = load_raw_edf(edf_f, verbose=verbose)
            trials, labels = edf_to_numpy(edf, mapping=mapping, length=length)

            for i, (trial, label) in enumerate(zip(trials, labels)):
                fname = f'{subject_id_run}_{i}.npy'
                if length and trial.shape[1] != length:
                    print(f'{fname} has invalid length of {trial.shape[1]}, not saving trial')
                else:
                    # np.savetxt(data_path / fname, np.float32(trial), delimiter=',')
                    np.save(data_path / fname, trial)
                    annotation_df_rows.append([str(edf_f), subject_id, fname, label])
        annotation_df = pd.DataFrame(annotation_df_rows, columns=['edf_file', 'subject_id', 'csv_file', 'label'])
        annotation_df.to_csv(data_path / 'annotations.csv')
    else:
        print('annotations.csv file found')
        annotation_df = pd.read_csv(data_path / 'annotations.csv')

    if not (data_path / 'train').is_dir() or not (data_path / 'test').is_dir():
        print('train_annotations.csv or test_annotations.csv not present, generating training/testing split')
        gen_train_test_dirs(data_dir)

    if not (data_path / 'train_mean.npy').is_file() or not (data_path / 'train_std.npy').is_file():
        print('normalizing')
        normalize_from_train(data_dir)

    return load_train_test(data_dir)

def gen_train_test_dirs(data_dir):
    '''
    DOES NOT SUPPORT WITHIN SUBJECTS YET
    '''
    data_path = Path(data_dir)
    shuffled_df = pd.read_csv(data_path / 'annotations.csv').sample(frac=1, random_state=0).reset_index(drop=True)

    df_len = len(shuffled_df)
    train_i, val_i, test_i = int(.7*df_len), int(.85*df_len), df_len

    # group_dfs = ['train': annotation_df[:train_i], 'val': annotation_df[train_i:val_i], 'test': annotation_df[val_i:]]
    group_labels = np.concatenate((np.repeat('train', train_i), np.repeat('val', (val_i - train_i)), np.repeat('test', (test_i - val_i))))
    shuffled_df['group'] = group_labels
    shuffled_df.to_csv(data_path / 'annotations.csv')

    groups = [shuffled_df[:train_i], shuffled_df[train_i:val_i], shuffled_df[val_i:]]
    # group_dfs = shuffled_df.iloc[:train], shuffled_df.iloc[train_i:val_i], shuffled_df.iloc[val_i:]

    for i, group in enumerate(['train', 'val', 'test']):
        group_path = data_path / group
        group_path.mkdir(parents=True, exist_ok=True)
        groups[i].apply(lambda x: shutil.move(str(data_path / x['csv_file']), str(group_path / x['csv_file'])), axis=1)
        groups[i].to_csv(group_path / 'annotations.csv')

def normalize_from_train(data_dir):
    data_path = Path(data_dir)
    groups = ['train', 'val', 'test']
    group_dfs = [pd.read_csv(data_path / group / 'annotations.csv') for group in groups]
    train_df = group_dfs[0]

    mean_sum = np.float64(np.repeat(0, 64))
    std_sum = np.float64(np.repeat(0, 64))
    for trial in train_df['csv_file']:
        ar = np.load(data_path / 'train' / trial)
        mean_sum += ar.mean(axis=1)
        std_sum += ar.std(axis=1)
    train_len = len(train_df['csv_file'])
    train_mean, train_std = (mean_sum / train_len).reshape(-1, 1), (std_sum / train_len).reshape(-1, 1)
    np.save(data_path / 'train_mean.npy', train_mean)
    np.save(data_path / 'train_std.npy', train_std)

    for df, group in zip(group_dfs, groups):
        for trial in df['csv_file']:
            np_path = data_path / group / trial
            ar = np.load(np_path)
            ar = (ar - train_mean) / train_std
            np.save(np_path, np.float32(ar))

def load_train_test(data_dir):
    data_path = Path(data_dir)

    return [eegDataset(data_path / group) for group in ['train', 'val', 'test']]
