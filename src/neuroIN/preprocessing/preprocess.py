


def preprocess(data_dir, targ_dir):
    orig_path, targ_path = Path(orig_dir), Path(targ_dir)

    assert orig_path.is_dir(), 'orig_dir must be a directory'
    targ_path.mkdir(parents=True, exist_ok=True)