from neuroIN.io import dataset

from pathlib import Path
import numpy as np

def trim_length(data_dir, length):
    """Function to trim all samples in a directory to the same length

    Any samples that are shorter than the specified length will be removed.

    :param data_dir: [description]
    :type data_dir: [type]
    :param length: [description]
    :type length: [type]
    """
    data_path = Path(data_dir)
    data = dataset.Dataset(data_dir)
    df = data.get_annotation_df()

    for np_f in data_path.rglob('*.npy'):
        sample = np.load(np_f)
        sample_len = sample.shape[-1]

        if sample_len < length:
            np_f.unlink()
            df = df[~df['np_file'].str.match(np_f.name)]
        else:
            sample = sample[..., :length]
            np.save(np_f, sample)
            df.loc[df['np_file'].str.match(np_f.name), 'length'] = length
    
    df.to_csv(data.data_path / "annotations.csv", index=False)