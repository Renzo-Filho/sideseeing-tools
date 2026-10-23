import os


def walk(dir_path):
    """Walk like os.walk, entering linked directories once so links cannot cause loops."""
    seen = set()

    for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
        real_root = os.path.realpath(root)

        if real_root in seen:
            dirnames[:] = []
            continue

        seen.add(real_root)
        yield root, dirnames, filenames


def get_dir_size(dir_path):
    """Calculate the total size of the directory in GB."""

    if not os.path.exists(dir_path):
        raise ValueError(f"Path '{dir_path}' does not exist.")

    if not os.path.isdir(dir_path):
        raise ValueError(f"Path '{dir_path}' is not a directory.")

    total_size = 0

    for dirpath, dirnames, filenames in walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)

            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size / (1024**3)
