from neuroIN.io.dataset import Dataset

from pathlib import Path
import numpy as np
from scipy.signal import hilbert
from scipy.stats import pearsonr
import torch
from mne.filter import filter_data
from tqdm import tqdm

def frequency_band_envelopes(data_dir, freq_bands, sfreq, k_temp, idxs=None, save=True):
    data_path = Path(data_dir)
    dataset = Dataset(data_path)
    
    data_shape = dataset[0][0].shape
    
    n_channels = data_shape[0]
    n_temp = data_shape[-1]
    
    
    if not idxs: idxs = list(range(len(dataset)))

    n_samples = len(idxs)

    freq_envs = np.zeros((len(freq_bands), n_samples, n_channels, (n_temp-k_temp)))
    
    for b, (low, high) in enumerate(freq_bands):
        print(f"Processing frequency band: [{low}-{high}]")
        for i in tqdm(idxs):
            data, _ = dataset[i]
            squared_amplitude_envelope = squared_envelope(data, sfreq, low, high)

            for k in range(n_temp-k_temp):
                freq_envs[b, i, :, k] = squared_amplitude_envelope[:, k:(k_temp+k)].mean(axis=-1)
    
    if save: np.save(data_path / 'freq_envs.npy', freq_envs)

    return freq_envs

def squared_envelope(data, sfreq, l_freq, h_freq):
    filtered_data = filter_data(data, sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)

    amplitude_envelope = np.abs(hilbert(filtered_data))
    squared_amplitude_envelope = np.square(amplitude_envelope)

    return squared_amplitude_envelope



def unit_outputs(data_dir, layer_weights, idxs=None, save=True):
    data_path = Path(data_dir)
    dataset = Dataset(data_path)

    data_shape = dataset[0][0].shape
    
    n_channels = data_shape[0]
    n_temp = data_shape[-1]

    n_filters = layer_weights.shape[0]
    k_temp = layer_weights.shape[-1]
    
    
    if not idxs: idxs = list(range(len(dataset)))

    n_samples = len(idxs)

    unit_outs = np.zeros((n_filters, n_samples, n_channels, (n_temp-k_temp)))

    for i in tqdm(idxs):
        data, _ = dataset[i]
        data = np.float32(data)
        for f in range(n_filters):
            filt = torch.tensor(layer_weights[f])
            for t in range(n_temp-k_temp):
                data_seg = torch.tensor(data[:, t:(k_temp+t)])
                unit_outs[f, i, :, t]  = filt.squeeze().unsqueeze(0).mm(data_seg.T).numpy()
    
    if save: np.save(data_path / 'unit_outs.npy', unit_outs)

    return unit_outs


def filter_corr(freq_envs, unit_outs):
    n_bands = freq_envs.shape[0]
    n_filts = unit_outs.shape[0]

    corrs = np.zeros((n_bands, n_filts))
    ps =  np.zeros((n_bands, n_filts))

    for f in range(n_filts):
        for b in range(n_bands):
            corr, p = pearsonr(freq_envs[b, :, :, :].flatten(), unit_outs[f, :, :, :].flatten())
            corrs[b, f] = corr
            ps[b, f] = p

    return corrs, ps