from os.path import splitext
from pathlib import Path

def dir_file_types(dir):
    """Generate a set containing all unique extensions in a directory

    :param dir: The directory to list files extensions from
    :type dir: string or pathlike
    :return: exts
    :rtype: set
    """
    data_path = Path(dir)
    exts = {ext for _, ext in [splitext(f) for f in data_path.rglob('*')]}
    return exts