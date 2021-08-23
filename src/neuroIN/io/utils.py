from os.path import splitext
from pathlib import Path

def dir_file_types(dir):
    """Generate a set containing all unique extensions in a directory

    :param dir: The directory to list files extensions from
    :type dir: string or pathlike
    :return: extensions
    :rtype: set
    """
    data_path = Path(dir)
    exts = {ext for _, ext in [splitext(f) for f in data_path.rglob('*')]}
    exts = {ext if ext[1:].isalpha() else '' for ext in exts}
    exts.discard('')
    return exts