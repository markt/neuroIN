from ..io.dataset import import_dataset, Dataset

import os
from pathlib import Path
from mne.datasets.eegbci import load_data

def import_eegmmidb(targ_dir, orig_dir=None, subjects='all', runs='all', download_only=False):
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
    if subjects == "all": subjects = [*range(1, 110)]
    if runs == "all": runs = [4, 8, 12]

    for subject in subjects:
        load_data(subject, runs, path=orig_dir)
    
    if not orig_dir:
        try:
            orig_dir = os.environ['MNE_DATASETS_EEGBCI_PATH']
        except KeyError:
            orig_dir = Path().home() / 'mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/'

    if not download_only: import_dataset(orig_dir, targ_dir, dataset_name="eegmmidb")

    Dataset(targ_dir).standardize_channels()