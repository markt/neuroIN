from ..io.dataset import import_dataset, Dataset

import os
from pathlib import Path
import numpy as np
import tqdm
from mne.datasets.eegbci import load_data
from mne.io import read_raw_edf
from mne import events_from_annotations

def edf_to_np(edf_f, mapping=None, ch_names=None, np_dtype=np.float32, resample_freq=None):
    """Loads a .EDF file into a list of NumPy arrays for events and labels.

    :param edf_f: The .EDF file to load from.
    :type edf_f: string or pathlib.Path
    :param ch_names: A dictionary mapping old channel names to new ones.
    :type ch_names: dict or callable
    :param np_dtype: Type of NumPy array
    :type np_dtype: type
    :return: Returns a list of NumPy arrays, an array of labels, a dict with mapping, and channel names
    """
    edf = read_raw_edf(edf_f, preload=True, verbose=False)
    if ch_names: edf.rename_channels(ch_names)

    if resample_freq:
        events, _ = events_from_annotations(edf, event_id=mapping, verbose=False)
        edf, _ = edf.resample(resample_freq, events=events)

    data, _ = edf[:, :]
    data = np_dtype(data)

    events, mapping = events_from_annotations(edf, event_id=mapping, verbose=False)
    time_samples, labels = events[:,0], events[:,2]
    if time_samples[0] == 0: time_samples = time_samples[1:] # do not split on first if it is zero

    data = np.split(data, time_samples, axis=1)
    data[-1] = data[-1][:,~np.all(data[-1] == 0, axis=0)] # trim zeros off of final event

    return data, labels, mapping, edf.ch_names

def import_eegmmidb(targ_dir, orig_dir=None, subjects='all', runs='all', download_only=False, mapping={'T0': -1, 'T1': 0, 'T2': 1}):
    """[summary]

    If specifying an original directory, users will be asked if they would like the directory as the default EEGBCI dataset path in the mne-python config

    :param targ_dir: [description]
    :type targ_dir: [type]
    :param orig_dir: [description], defaults to None
    :type orig_dir: [type], optional
    :param subjects: [description], defaults to 'all'
    :type subjects: str, optional
    :param runs: [description], defaults to 'all'
    :type runs: str, optional
    """
    targ_path = Path(targ_dir).expanduser()
    if orig_dir:
        orig_path = Path(orig_dir).expanduser()
    else:
        try:
            orig_path = Path(os.environ['MNE_DATASETS_EEGBCI_PATH'])
        except KeyError:
            orig_path = Path().home() / 'mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/'

    if subjects == "all": subjects = [*range(1, 110)]
    if runs == "all": runs = [4, 8, 12]

    print("Downloading dataset...")
    for subject in tqdm.tqdm(subjects):
        load_data(subject, runs, path=str(orig_path), verbose=False)
    print("Dataset downloaded.")

    if not download_only:
        dataset_extenions = {'.edf': edf_to_np}
        import_dataset(orig_path, targ_path, dataset_extenions=dataset_extenions, dataset_name="eegmmidb", mapping=mapping)

        data = Dataset(targ_path)

        data.standardize_channels()