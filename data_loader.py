import os

import numpy as np

from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from mne import events_from_annotations
from mne.channels.montage import DigMontage


def load_raw_edf(fname, ch_names=None, montage_kind='standard_1005'):
    '''
    Loads a trial by filename and returns an MNE RawEDF object

    ch_names is an optional dictionary for renaming channels
    montage_kind is an optional DigMontage or montage_kind string
    '''
    raw = read_raw_edf(fname, preload=True)

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





class Trials():
    '''
    Custom class to store raw data from a trial
    '''

    def __init__(self, fname):
        try:
            raw = read_raw_edf(fname, preload=True)

        except(OSError, FileNotFoundError):
            print(f'Unable to find {fname}')

        except Exception as error:
            print(f'Error: {error}')

        # replace channel names
        new_names = {ch: ch.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp') for ch in raw.ch_names}
        raw.rename_channels(new_names)

        # standardize montage
        montage = make_standard_montage('standard_1005')
        raw.set_montage(montage)



        events, blank = events_from_annotations(raws)

        data, times = raw[:, :]
        self.data, self.times = data, times

        trials = []
        labels = []

        mapping = {1:'T0', 2:'T1', 3:'T2'}

        for i in range(len(events) - 1):
            start, end, label = events[i][0], events[i+1][0], mapping[events[i][2]]
            trials.append(data[:,start:end])
            labels.append(label)
            
        start, end, label = events[-1][0], data.shape[1], mapping[events[-1][2]]
        trials.append(data[:,start:end])
        labels.append(label)

        self.trials = trials
        self.labels = labels