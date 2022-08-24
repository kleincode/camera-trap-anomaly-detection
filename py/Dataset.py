import os
from tqdm import tqdm
from py.DatasetStatistics import DatasetStatistics
from py.FileUtils import list_folders, list_jpegs_recursive, expected_subfolders, verify_expected_subfolders
from py.Session import Session

# Represents the whole dataset consisting of multiple sessions. Can be used to get
# session instances or to get an statistics instance.
class Dataset:

    def __init__(self, base_path: str):
        """Create a new dataset instance.

        Args:
            base_path (str): Path to dataset, should contain subfolders for sessions.
        """
        self.base_path = base_path
        self.raw_sessions = []
        self.__parse_subdirectories()
    

    def __parse_subdirectories(self):
        self.raw_sessions = sorted(list_folders(self.base_path))
        # Verify every session contains the subfolders Motion, Lapse, Full
        for folder in self.raw_sessions:
            path = os.path.join(self.base_path, folder)
            verify_expected_subfolders(path)
        print(f"Found {len(self.raw_sessions)} sessions")


    def get_sessions(self) -> list:
        """Get names of all sessions (without prefixes).

        Returns:
            list of str: session names
        """
        # cut off the first 33 characters (redundant)
        return [name[33:] for name in self.raw_sessions]
    
    def create_statistics(self) -> DatasetStatistics:
        """Accumulate statistics over the dataset and return a new statistics instance.

        Returns:
            DatasetStatistics: statistics instance
        """
        counts = {}
        for folder in tqdm(self.raw_sessions):
            counts[folder[33:]] = {}
            counts[folder[33:]]["Total"] = 0
            for subfolder in expected_subfolders:
                path = os.path.join(self.base_path, folder, subfolder)
                numFiles = len(list_jpegs_recursive(path))
                counts[folder[33:]][subfolder] = numFiles
                counts[folder[33:]]["Total"] += numFiles
        return DatasetStatistics(counts)

    def create_session(self, session_name: str) -> Session:
        """Return a new session instance from the session name.

        Args:
            session_name (str): Session name, e.g. beaver_01. Not case-sensitive.

        Raises:
            ValueError: No or multiple sessions matching session name

        Returns:
            Session: Session instance
        """
        if session_name in self.raw_sessions:
            return Session(os.path.join(self.base_path, session_name))
        filtered = [s for s in self.raw_sessions if session_name.lower() in s.lower()]
        if len(filtered) == 0:
            raise ValueError(f"There are no sessions matching this name: {filtered}")
        elif len(filtered) > 1:
            raise ValueError(f"There are several sessions matching this name: {session_name}")
        return Session(os.path.join(self.base_path, filtered[0]))
