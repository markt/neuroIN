from ..io.dataset import import_dataset, Dataset

from pathlib import Path
import numpy as np
from mne.io import read_raw_gdf
from mne import events_from_annotations
from mne.preprocessing import regress_artifact

def gdf_to_np(gdf_f, mapping={"768": -1, "769": 0, "770": 1, "771": 2, "772": 3}, ch_names=None, np_dtype=np.float32, resample_freq=None, clean=True):
    """Loads a .EDF file into a list of NumPy arrays for events and labels.

    :param edf_f: The .EDF file to load from.
    :type edf_f: string or pathlib.Path
    :param ch_names: A dictionary mapping old channel names to new ones.
    :type ch_names: dict or callable
    :param np_dtype: Type of NumPy array
    :type np_dtype: type
    :return: Returns a list of NumPy arrays, an array of labels, a dict with mapping, and channel names
    """
    gdf = read_raw_gdf(gdf_f, preload=True, verbose=False)

    if clean:
        gdf.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})
        gdf, _ = regress_artifact(gdf)

    if resample_freq:
        events, _ = events_from_annotations(gdf, event_id=mapping, verbose=False)
        gdf, _ = gdf.resample(resample_freq, events=events)

    if ch_names: gdf.rename_channels(ch_names)
    ch_names = gdf.ch_names

    data, _ = gdf[:, :]
    data = np_dtype(data)

    if clean:
        data = data[:22]
        ch_names = ch_names[:22]

    start_times = events_from_annotations(gdf, event_id={"768": -1}, verbose=False)[0][:,0]
    data = np.split(data, start_times, axis=1)[1:] #first split is before first trial

    mapping = mapping
    labels = events_from_annotations(gdf, event_id=mapping, verbose=False)[0][:,2]

    return data, labels, mapping, ch_names

def import_bci_iv_2a(targ_dir, orig_dir):
    targ_path, orig_path = Path(targ_dir).expanduser(), Path(orig_dir).expanduser()
    dataset_extenions = {'.gdf': gdf_to_np}
    import_dataset(orig_path, targ_path, dataset_extenions=dataset_extenions, dataset_name="bci_iv_2a")