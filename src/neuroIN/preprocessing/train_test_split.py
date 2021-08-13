import shutil
from pathlib import Path
import numpy as np
import pandas as pd

def create_split(data_dir, train_prop=0.7, val_prop=0.15, by_subject=False):
    assert (train_prop + val_prop) > 0 and (train_prop + val_prop) < 1, "Sum of training and validation proportion must be greater than zero and less than one"

    data_path = Path(data_dir)

    # shuffle all samples
    annotations_df = pd.read_csv(data_path / 'annotations.csv')
    shuffled_df = annotations_df.sample(frac=1)

    # randomly split samples into sets
    if not by_subject:
        # using length of dataframe, split by training/val/test proportions
        df_len = len(shuffled_df)
        train_i, val_i, test_i = int(train_prop*df_len), int((train_prop+val_prop)*df_len), df_len

        # generate group labels
        group_labels = np.concatenate((np.repeat('train', train_i), np.repeat('val', (val_i - train_i)), np.repeat('test', (test_i - val_i))))

    # otherwise, randomly split subjects into sets
    else:
        # get an array with unique subject IDs
        ids = shuffled_df['subject_id'].unique()

        # using length of subject ID array, split by training/val/test proportions
        ids_len = len(ids)
        train_i, val_i, test_i = int(train_prop*ids_len), int((train_prop+val_prop)*ids_len), ids_len

        # get subject labels and dict from ID to group
        subject_labels = np.concatenate((np.repeat('train', train_i), np.repeat('val', (val_i - train_i)), np.repeat('test', (test_i - val_i))))
        id_to_group = {ids[i]: subject_labels[i] for i in range(ids_len)}

        # generate group labels
        group_labels = shuffled_df['subject_id'].apply(lambda subject: id_to_group[subject])

    # add group labels to annotations file and save
    shuffled_df['group'] = group_labels
    shuffled_df.to_csv(data_path / 'annotations.csv')

    # make directories for sets and set specific annotation files
    for group in ['train', 'val', 'test']:
        group_path = data_path / group
        group_path.mkdir(parents=True, exist_ok=True)
        shuffled_df[shuffled_df['group'].str.match(group)].to_csv(group_path / 'annotations.csv')

    # move each file into its groups' directory
    shuffled_df.apply(lambda x: shutil.move(str(data_path / x['np_file']), str(data_path / x['group'] / x['np_file'])), axis=1)