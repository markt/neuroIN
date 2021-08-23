from neuroIN.io import dataset

from pathlib import Path
import numpy as np

def trim_length(data_dir, length, start_i=0):
    """Function to trim all samples in a Dataset directory to the same length

    Any samples that are shorter than the specified length will be removed.

    :param data_dir: the directory of the Dataset
    :type data_dir: str or pathlike
    :param length: the length to trim to
    :param start_i: the index to start trimming from, defaults to 0
    :type start_i: int, optional
    """
    data_path = Path(data_dir)
    data = dataset.Dataset(data_dir)
    df = data.get_annotation_df()

    for np_f in data_path.rglob('*.npy'):
        sample = np.load(np_f)
        sample_len = sample.shape[-1]

        if (sample_len-start_i) < length:
            print(f"{np_f} is too short!")
            np_f.unlink()
            df = df[~df['np_file'].str.match(np_f.name)]
        else:
            sample = sample[..., start_i:(start_i+length)]
            np.save(np_f, sample)
            df.loc[df['np_file'].str.match(np_f.name), 'length'] = length
    
    df.to_csv(data.data_path / "annotations.csv", index=False)