import os
from tqdm import tqdm
from py.DatasetStatistics import DatasetStatistics
from py.FileUtils import list_folders, list_jpegs_recursive

expected_subfolders = sorted(["Motion", "Lapse", "Full"])

class Dataset:

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.raw_sessions = []
        self.__parse_subdirectories()
    

    def __parse_subdirectories(self):
        self.raw_sessions = sorted(list_folders(self.base_path))
        # Verify every session contains the subfolders Motion, Lapse, Full
        for folder in self.raw_sessions:
            path = os.path.join(self.base_path, folder)
            subfolders = list_folders(path)
            assert sorted(subfolders) == expected_subfolders
        print(f"Found {len(self.raw_sessions)} sessions")


    def get_sessions(self) -> list:
        # cut off the first 33 characters (redundant)
        return [name[33:] for name in self.raw_sessions]
    
    def create_statistics(self) -> DatasetStatistics:
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
