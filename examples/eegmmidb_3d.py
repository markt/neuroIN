import os
from pathlib import Path

from neuroIN.datasets import eegmmidb
from neuroIN.io import dataset

# path data will be imported into
data_path = Path(os.getcwd()) / "data/eegmmidb_3d"
orig_path = Path(os.getcwd()) / "data/orig_eegmmidb"
eegmmidb.import_eegmmidb(data_path, orig_dir=orig_path)

# load Dataset object
data = dataset.Dataset(data_path)

# preprocesses
data.trim_data_length(640)
data.create_data_split(train_prop=.8, val_prop=0)
data.normalize_data_from_train()
data.data_to_3d()