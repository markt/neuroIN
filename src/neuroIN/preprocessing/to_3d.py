from pathlib import Path
import numpy as np

def dir_to_3d(data_dir, type_3d='d', ch_names=None, d_idxs_override=None):
    """Convert all trials in the Dataset directory into 3D format

    :param data_dir: the directory of the Dataset
    :type data_dir: str or pathlike
    :param type_3d: the type of 3D configuration to use, defaults to 'd'
    :type type_3d: str, optional
    :param ch_names: channel names to use for finding indices to keep, defaults to None
    :type ch_names: list, optional
    :param d_idxs_override: indices for D configuration that can be used to override the automatic index selection, defaults to None
    :type d_idxs_override: list, optional
    """
    data_path = Path(data_dir)

    if not ch_names:
        ch_names = [
            'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1',
            'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8',
            'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1',
            'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
            'O1', 'Oz', 'O2', 'Iz'
        ]
    ch_names_to_idx = {name: ch_names.index(name) for name in ch_names}

    if type_3d == 'd':
        d_names = ['FCz', 'C2', 'CP4', 'Cz', 'CP2',
                   'C1', 'CPz', 'P2', 'CP1', 'Pz',
                   'CP3', 'P1', 'POz']
        if not d_idxs_override:
            assert set(d_names).issubset(ch_names), "Dataset's channel names must be standardized!"

            d_idxs = [ch_names_to_idx[n] for n in d_names]
        else:
            d_idxs = d_idxs_override
        to_3d_func = npy_to_d_3d

    for f in data_path.rglob('*.npy'):
        to_3d_func(f, d_idxs, save=True)

def npy_to_d_3d(npy_file, d_idxs, save=True):
    data = np.load(npy_file)
    d_type = data[d_idxs,:]

    # insert zeros between channels so shape comes out 5x5
    zero_idxs = [i for i in range(1, 13)]
    d_type = np.insert(d_type, zero_idxs, 0, axis=0)

    d_type_3d = np.reshape(d_type, (5, 5, -1))
    if save: np.save(npy_file, d_type_3d)
    
    return d_type_3d