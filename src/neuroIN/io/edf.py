import numpy as np
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