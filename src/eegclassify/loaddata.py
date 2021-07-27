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
from mne.datasets import eegbci

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

        try:
            self.weights = pd.read_csv(self.data_dir / 'weights.csv')
        except FileNotFoundError:
            self.weights = None

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


def load_edf_to_numpy(edf_f, mapping, length=None, verbose=False):
    edf = load_raw_edf(edf_f, verbose=verbose)
    return edf_to_numpy(edf, mapping=mapping, length=length)

def load_experiment(data_dir: str, orig_dir: str, mapping: dict, force_process: bool=False, orig_f_ext: str='.edf', orig_f_pattern: str='*', to_np_func=load_edf_to_numpy, between_subjects: bool=True, train_with_null=False, norm_by_pixel=False, n_channels=64, length: int=None, verbose: bool=False):
    '''
    Function to load an experiment
    Data is first processed and saved into NumPy binary files (if processed files do not already reside in data_dir),
    then processed files are organized into training/validation/testing directories,
    then all data is normalized (to mean=0, std=1) based on the training data,
    finally a list of eegDataset objects is returned

    Parameters
    ----------
    data_dir : str
        the directory processed data does/will reside
    orig_dir : str
        the directory unprocessed data resides (only used if annotations.csv is not present in data_dir)
    mapping : dict
        the mapping from string labels to integers for classification (integers should start at 0 and increase monotonically)
    force_process : bool
        force data to be reprocessed even if it already has been
    orig_f_ext : string
        the extension of original, unprocessed data files
    orig_f_pattern : string
        the pattern of original filenames
    to_np_func: the function used to convert
    between_subjects: whether to use between or within subjects for training/testing split
    length: the length that each trial should be (longer trials will be truncated while shorter trials will not be processed)
    verbose: whether to unsuppress optional outputs
    '''
    data_path = Path(data_dir)
    assert data_path.is_dir(), 'data_dir must be a directory'

    if between_subjects: assert not train_with_null, 'train_with_null can only be used (set to True) if between_subjects is not used (set to False)'

    # convert original
    annotation_path = data_path / 'annotations.csv'
    if not annotation_path.is_file() or force_process:
        # if annotations.csv is not present, data is processed from original files
        if verbose: print('annotations.csv file not found, generating NumPy files from original data...')

        orig_path = Path(orig_dir)
        assert orig_path.is_dir(), 'orig_dir must be a directory'

        annotation_df_rows = []
        for orig_f in orig_path.rglob(orig_f_pattern + orig_f_ext):
            orig_fname = str(orig_f).split('/')[-1].split('.')[0]
            subject_id = re.search(r'\d+', orig_fname).group()

            trials, labels = to_np_func(orig_f,  mapping=mapping, length=length, verbose=verbose)

            for i, (trial, label) in enumerate(zip(trials, labels)):
                fname = f'{orig_fname}_{i}.npy'
                if length and trial.shape[1] != length:
                    if verbose: print(f'{fname} has invalid length of {trial.shape[1]}, not saving trial')
                else:
                    np.save(data_path / fname, trial)
                    annotation_df_rows.append([str(orig_f.resolve()), subject_id, fname, label])
        annotation_df = pd.DataFrame(annotation_df_rows, columns=['orig_file', 'subject_id', 'csv_file', 'label'])
        annotation_df.to_csv(data_path / 'annotations.csv')
    else:
        if verbose: print('annotations.csv file found')
        annotation_df = pd.read_csv(data_path / 'annotations.csv')

    # generate training/validation/testing split
    if not (data_path / 'train').is_dir() or not (data_path / 'test').is_dir():
        print('train_annotations.csv or test_annotations.csv not present, generating training/testing split')
        gen_train_test_dirs(data_dir, between_subjects=between_subjects, train_with_null=train_with_null)

    # normalize data
    if not (data_path / 'train_mean.npy').is_file() or not (data_path / 'train_std.npy').is_file():
        print('normalizing')
        if norm_by_pixel:
            normalize_from_train_pixel(data_dir, n_channels=n_channels)
        else:
            normalize_from_train(data_dir, n_channels=n_channels)

    return load_train_test(data_dir)









def load_eegmmidb(data_dir, edf_dir=None, between_subjects=True, train_with_null=False, mapping={'T0': 0, 'T1': 1, 'T2': 2}, norm_by_pixel=False, length=640, verbose=False):
    data_path = Path(data_dir)
    assert data_path.is_dir(), 'parent_dir must be a directory'

    annotation_path = data_path / 'annotations.csv'
    if not annotation_path.is_file():
        if not edf_dir:
            try:
                edf_dir = os.environ['MNE_DATASETS_EEGBCI_PATH']
                edf_path = Path(edf_dir)
            except KeyError:
                if verbose: print('MNE_DATASETS_EEGBCI_PATH not set, using default location ~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/')
                edf_path = Path().home() / 'mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/'
        else:
            edf_path = Path(edf_dir)
    else:
        edf_path = None
    
    return load_experiment(data_dir, edf_path, mapping=mapping, between_subjects=between_subjects, train_with_null=train_with_null, norm_by_pixel=norm_by_pixel, length=length, verbose=verbose)



def load_bci_iv_2a(data_dir, npz_dir=None, between_subjects=True, train_with_null=False, mapping={769: 0, 770: 1, 771: 2, 772: 3}, n_channels=22, length=1875, verbose=False):
    '''
    d
    '''
    start_id = 768
    last_channel = n_channels

    def npz_to_np(npz_f,  mapping, length=length, verbose=verbose):
        data = np.load(npz_f)

        raw = data['s'].T
        etyp = data['etyp'].T[0]
        epos = data['epos'].T[0]
        edur = data['edur'].T[0]
        artifacts = data['artifacts'].T[0]

        trial_idxs = [i for i, event in enumerate(etyp) if event == start_id]

        trials = []
        labels = []

        for i in trial_idxs:
            try:
                labels.append(mapping[etyp[i+1]])
                
                event_dur = length if length else edur[i]
                start, stop = epos[i], (epos[i]+event_dur)
                trials.append(raw[:last_channel, start:stop])
            except KeyError:
                if verbose: print(f'rejected trial (etyp={etyp[i+1]})')
                continue

        return trials, labels

    return load_experiment(data_dir, npz_dir, mapping=mapping, orig_f_ext='.npz', to_np_func=npz_to_np, between_subjects=between_subjects, train_with_null=train_with_null, n_channels=n_channels, length=length, verbose=verbose)




def load_inria_bci_challenge(data_dir, csv_dir, between_subjects=True, mapping={0: 0, 1: 1}, n_channels=56, length=260, verbose=False):
    orig_f_pattern = 'Data*'
    orig_f_ext = '.csv'
    
    sample_rate = 200
    # TODO: implement parameter HERE FOR LENGTH BASED ON DURATION?? (e.g. length = 1.3 * sample_rate)
    
    def csv_to_np(csv_f, mapping, length=length, verbose=verbose):
        df = pd.read_csv(Path(data_dir) / csv_f)
        data = df.iloc[:,1:n_channels+1].values.T
        idxs = [i for i, num in enumerate(df['FeedBackEvent']) if num == 1]
        
        train_labels = pd.read_csv(Path(orig_dir) / 'TrainLabels.csv')
        test_labels = pd.read_csv(Path(orig_dir) / 'SampleSubmission.csv')
        true_test_labels = pd.read_csv(Path(orig_dir) / 'true_labels.csv', header=None).values
        test_labels['Prediction'] = test_labels
        all_labels = pd.concat([train_labels, sample_df])
        
        trials = []
        labels = []
        
        subject_id, session = str(csv_f).split('/')[-1].split('.')[0].split('_')[1:]
        
        for i, idx in enumerate(idxs):
            label = all_labels[all_labels['IdFeedBack'] == f'{subject_id}_{session}_FB{(i+1):03d}']['Prediction'].item()
            labels.append(label)
            trials.append(data[:, idx:idx+length])
        return trials, labels
    
    return load_experiment(data_dir, csv_dir, mapping=mapping, to_np_func=csv_to_np, orig_f_pattern=orig_f_pattern, orig_f_ext=orig_f_ext, n_channels=n_channels, length=length, verbose=verbose)








def gen_train_test_dirs(data_dir, between_subjects=True, train_with_null=False):
    '''
    Currently train_with_null is only available for between_subjects (the treatment really does not make much sense with within)

    train_with_null moves all null/neutral/resting trials into the training set
    between subjects must be False
    '''
    data_path = Path(data_dir)
    shuffled_df = pd.read_csv(data_path / 'annotations.csv').sample(frac=1, random_state=0).reset_index(drop=True)

    if between_subjects:
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
    else:
        ids = shuffled_df['subject_id'].unique()
        ids_length = len(ids)

        train_i, val_i, test_i = int(.7*ids_length), int(.85*ids_length), ids_length
        group_labels = np.concatenate((np.repeat('train', train_i), np.repeat('val', (val_i - train_i)), np.repeat('test', (test_i - val_i))))
        ids_dict = {ids[i]: group_labels[i] for i in range(ids_length)}
        shuffled_df['group'] = shuffled_df['subject_id'].apply(lambda x: ids_dict[x])

        if train_with_null:
            shuffled_df['group'] = shuffled_df.apply(lambda r: 'train' if r['label'] == 0 else r['group'], axis=1)

        shuffled_df.to_csv(data_path / 'annotations.csv')

        for group in ['train', 'val', 'test']:
            group_df = shuffled_df[shuffled_df['group'] == group]
            group_path = data_path / group
            group_path.mkdir(parents=True, exist_ok=True)
            group_df.apply(lambda x: shutil.move(str(data_path / x['csv_file']), str(group_path / x['csv_file'])), axis=1)
            group_df.to_csv(group_path / 'annotations.csv')



def normalize_from_train_pixel(data_dir, inv_weighted=False, n_channels=64):
    data_path = Path(data_dir)
    groups = ['train', 'val', 'test']
    group_dfs = [pd.read_csv(data_path / group / 'annotations.csv') for group in groups]
    train_df = group_dfs[0]
    
    channel_means = np.zeros((n_channels, 640, len(train_df)))
    for i, csv in enumerate(train_df['csv_file']):
        ar = np.load(data_path / 'train' / csv)
        channel_means[:, :, i] = ar
    
    if inv_weighted:
        inv_weight_dict = (1 / train_df['label'].value_counts(normalize=True)).to_dict()
        weights = [inv_weight_dict[label] for label in train_df['label']]
    else:
        weights = None

    train_mean = np.average(channel_means, weights=None, axis=2)
    train_std = np.average((channel_means - train_mean[:,:,None])**2, weights=None, axis=2)
    np.save(data_path / 'train_mean.npy', train_mean)
    np.save(data_path / 'train_std.npy', train_std)
    
    for df, group in zip(group_dfs, groups):
        for trial in df['csv_file']:
            np_path = data_path / group / trial
            ar = np.load(np_path)
            ar = (ar - train_mean) / train_std
            np.save(np_path, np.float32(ar)) # save to float32 so smaller for PyTorch


def normalize_from_train(data_dir, inv_weighted=False, n_channels=64):
    data_path = Path(data_dir)
    groups = ['train', 'val', 'test']
    group_dfs = [pd.read_csv(data_path / group / 'annotations.csv') for group in groups]
    train_df = group_dfs[0]
    
    channel_means = np.zeros((n_channels, len(train_df)))
    for i, csv in enumerate(train_df['csv_file']):
        ar = np.load(data_path / 'train' / csv)
        channel_means[:, i] = ar.mean(axis=1)
    
    if inv_weighted:
        inv_weight_dict = (1 / train_df['label'].value_counts(normalize=True)).to_dict()
        weights = [inv_weight_dict[label] for label in train_df['label']]

        weight_tensor = torch.Tensor(list(inv_weight_dict.values()))
        torch.save(weight_tensor, data_path / 'weights.pt')
    else:
        weights = None

        
    train_mean = np.average(channel_means, weights=weights, axis=1).reshape(-1, 1)
    train_std = np.average((channel_means - avg.reshape(-1, 1))**2, weights=weights, axis=1).reshape(-1, 1)
    np.save(data_path / 'train_mean.npy', train_mean)
    np.save(data_path / 'train_std.npy', train_std)
    
    for df, group in zip(group_dfs, groups):
        for trial in df['csv_file']:
            np_path = data_path / group / trial
            ar = np.load(np_path)
            ar = (ar - train_mean) / train_std
            np.save(np_path, np.float32(ar)) # save to float32 so smaller for PyTorch

def load_train_test(data_dir):
    data_path = Path(data_dir)

    return [eegDataset(data_path / group) for group in ['train', 'val', 'test']]
