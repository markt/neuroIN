import numpy as np
from pathlib import Path
from mne.io import read_raw_edf
from mne import events_from_annotations

def edf_to_np(edf_f, mapping=None, ch_names=None, np_dtype=np.float32):
    """Loads a .EDF file into a list of NumPy arrays for events and labels.

    :param edf_f: The .EDF file to load from.
    :type edf_f: string or pathlib.Path
    :param ch_names: A dictionary mapping old channel names to new ones.
    :type ch_names: dict or callable
    :param np_dtype: Type of NumPy array
    :type np_dtype: type
    :return: Returns a list of NumPy arrays, an array of labels, and a dict with mapping
    """
    edf = read_raw_edf(edf_f, preload=True)
    if ch_names: edf.rename_channels(ch_names)

    data, times = edf[:, :]
    data = np_dtype(data)
    data = np.split(data, times, axis=1)[1:] # avoid first event of zero length

    events, mapping = events_from_annotations(edf, event_id=mapping)
    labels = events[:,2]

    return data, labels, mapping