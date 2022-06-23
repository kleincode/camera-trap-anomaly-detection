from glob import glob
import os
import pickle

expected_subfolders = sorted(["Motion", "Lapse", "Full"])

def list_folders(path: str) -> list:
    """Returns the names of all immediate child folders of path.

    Args:
        path (str): path to search

    Returns:
        list: list of all child folder names
    """
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def list_jpegs_recursive(path: str) -> list:
    """Recursively lists all jpeg files in path.

    Args:
        path (str): path to search

    Returns:
        list: list of all jpeg files
    """
    return [name for name in glob(os.path.join(path, "**/*.jpg"), recursive=True) if os.path.isfile(os.path.join(path, name))]

def verify_expected_subfolders(session_path: str):
    """Assert that the given session folder contains exactly the three subfolders Motion, Lapse, Full.

    Args:
        session_path (str): session folder path
    """
    subfolders = list_folders(session_path)
    if sorted(subfolders) != sorted(expected_subfolders):
        raise AssertionError(f"{session_path}: Expected subfolders {expected_subfolders} but found {subfolders}")

# Pickle helpers

def dump(filename: str, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)