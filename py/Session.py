from datetime import datetime
import pickle
import subprocess
from warnings import warn
import os
from tqdm import tqdm

from py.FileUtils import list_folders, list_jpegs_recursive, verify_expected_subfolders
from py.ImageUtils import get_image_date

class Session:
    def __init__(self, folder: str):
        self.folder = folder
        # session name = folder name[33:], the first 33 characters are always the same
        self.name = os.path.basename(folder)[33:]
        print(f"Session '{self.name}' at folder: {self.folder}")
        assert self.name != ""
        verify_expected_subfolders(self.folder)
        self.scanned = False
        # maps lapse files to their exif dates (for statistic and prediction purposes)
        self.lapse_dates = {}
        # maps motion files to their exif dates (for statistic purposes)
        self.motion_dates = {}
        # maps exif dates to lapse files (for prediction purposes)
        self.lapse_map = {}
        self.load_scans()
        if not self.scanned:
            print("Session not scanned. Run session.scan() to create scan files")
    
    def load_scans(self):
        lapse_dates_file = os.path.join("session_scans", self.name, "lapse_dates.pickle")
        motion_dates_file = os.path.join("session_scans", self.name, "motion_dates.pickle")
        lapse_map_file = os.path.join("session_scans", self.name, "lapse_map.pickle")
        lapse_dates_exists = os.path.isfile(lapse_dates_file)
        motion_dates_exists = os.path.isfile(motion_dates_file)
        lapse_map_exists = os.path.isfile(lapse_map_file)
        if lapse_dates_exists and motion_dates_exists and lapse_map_exists:
            with open(lapse_dates_file, "rb") as handle:
                self.lapse_dates = pickle.load(handle)
            with open(motion_dates_file, "rb") as handle:
                self.motion_dates = pickle.load(handle)
            with open(lapse_map_file, "rb") as handle:
                self.lapse_map = pickle.load(handle)
            self.scanned = True
            print("Loaded scans.")
        else:
            if not (not lapse_dates_exists and not motion_dates_exists and not lapse_map_exists):
                warn(f"Warning: Only partial scan data available. Not loading.")
            self.scanned = False
    
    def save_scans(self):
        os.makedirs(os.path.join("session_scans", self.name), exist_ok=True)
        lapse_dates_file = os.path.join("session_scans", self.name, "lapse_dates.pickle")
        motion_dates_file = os.path.join("session_scans", self.name, "motion_dates.pickle")
        lapse_map_file = os.path.join("session_scans", self.name, "lapse_map.pickle")
        with open(lapse_dates_file, "wb") as handle:
            pickle.dump(self.lapse_dates, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {lapse_dates_file}")
        with open(motion_dates_file, "wb") as handle:
            pickle.dump(self.motion_dates, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {motion_dates_file}")
        with open(lapse_map_file, "wb") as handle:
            pickle.dump(self.lapse_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {lapse_map_file}")
    
    def scan(self, force=False, auto_save=True):
        if self.scanned and not force:
            raise ValueError("Session is already scanned. Use force=True to scan anyway and override scan progress.")
        # Scan motion dates
        print("Scanning motion dates...")
        self.motion_dates = {}
        motion_folder = os.path.join(self.folder, "Motion")
        for file in tqdm(list_jpegs_recursive(motion_folder)):
            self.motion_dates[os.path.relpath(file, motion_folder)] = get_image_date(file)
        # Scan lapse dates
        print("Scanning lapse dates...")
        self.lapse_dates = {}
        lapse_folder = os.path.join(self.folder, "Lapse")
        for file in tqdm(list_jpegs_recursive(lapse_folder)):
            self.lapse_dates[os.path.relpath(file, lapse_folder)] = get_image_date(file)
        # Create lapse map
        print("Creating lapse map...")
        self.lapse_map = {}
        for file, date in self.lapse_dates.items():
            if date in self.lapse_map:
                self.lapse_map[date].append(file)
            else:
                self.lapse_map[date] = [file]
        self.scanned = True
        # Auto save
        if auto_save:
            print("Saving...")
            self.save_scans()
    
    def check_lapse_duplicates(self) -> bool:
        total = 0
        total_duplicates = 0
        total_multiples = 0
        deviant_duplicates = []
        for date, files in tqdm(self.lapse_map.items()):
            total += 1
            if len(files) > 1:
                total_duplicates += 1
                file_size = -1
                for f in files:
                    f_size = os.path.getsize(os.path.join(self.folder, "Lapse", f))
                    if file_size == -1:
                        file_size = f_size
                    elif f_size != file_size:
                        deviant_duplicates.append(date)
                        break
                if len(files) > 2:
                    total_multiples += 1
        deviant_duplicates.sort()
        print(f"* {total} lapse dates")
        print(f"* {total_duplicates} duplicates")
        print(f"* {total_multiples} multiples (more than two files per date)")
        print(f"* {len(deviant_duplicates)} deviant duplicates: {deviant_duplicates}")
        return total, total_duplicates, total_multiples, deviant_duplicates
    
    def open_images_for_date(self, date: datetime):
        img_names = self.lapse_map.get(date, [])
        if len(img_names) == 0:
            warn("No images for this date!")
        for i, img_name in enumerate(img_names):
            full_path = os.path.join(self.folder, "Lapse", img_name)
            print(f"#{i+1} {full_path}")
            subprocess.call(("xdg-open", full_path))
