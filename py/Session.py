from datetime import datetime, timedelta
import pickle
import random
import cv2 as cv
import subprocess
from warnings import warn
import os
from tqdm import tqdm
import matplotlib.image as mpimg
from skimage import transform, io
import IPython.display as display

from py.FileUtils import list_folders, list_jpegs_recursive, verify_expected_subfolders
from py.ImageUtils import display_images, get_image_date

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
        # maps exif dates to motion files (for csv mapping purposes, generated on demand)
        self.motion_map = None
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

    def get_motion_image_from_filename(self, filename: str) -> "MotionImage":
        if filename in self.motion_dates:
            return MotionImage(self, filename, self.motion_dates[filename])
        else:
            raise ValueError(f"Unknown motion file name: {filename}")
    
    def __generate_motion_map(self):
        """Populates self.motion_map which maps dates to motion images
        """
        if self.motion_map is not None:
            return
        print("Generating motion map...")
        self.motion_map = {}
        for filename, date in self.motion_dates.items():
            if date in self.motion_map:
                self.motion_map[date].append(filename)
            else:
                self.motion_map[date] = [filename]
    
    def get_motion_images_from_date(self, date: datetime):
        self.__generate_motion_map()
        filenames = self.motion_map.get(date, [])
        return [MotionImage(self, filename, date) for filename in filenames]
    
    def get_random_motion_image(self, day_only=False, night_only=False) -> "MotionImage":
        if len(self.motion_dates) == 0:
            raise ValueError("No motion images in session!")
        img = None
        while img is None or (day_only and img.is_nighttime()) or (night_only and img.is_daytime()):
            filename = random.choice(list(self.motion_dates.keys()))
            img = MotionImage(self, filename, self.motion_dates[filename])
        return img
    
    def get_random_motion_image_set(self, day_only=False, night_only=False) -> list:
        """Returns a list of all motion images with the same date +- 10 min.
        The date is picked randomly from all available dates.
        May loop indefinitely if there are no matching motion images.

        Args:
            day_only (bool, optional): Only pick daytime images. Defaults to False.
            night_only (bool, optional): Only pick nighttime images. Defaults to False.

        Raises:
            ValueError: No motion images in session

        Returns:
            list: Non-empty list of motion images with the same date
        """
        self.__generate_motion_map()
        if len(self.motion_map) == 0:
            raise ValueError("No motion images in session!")
        imgs = []
        date = None
        while len(imgs) == 0 or (day_only and imgs[0].is_nighttime()) or (night_only and imgs[0].is_daytime()):
            date = random.choice(list(self.motion_map.keys()))
            filenames = self.motion_map.get(date, [])
            imgs = [MotionImage(self, filename, date) for filename in filenames]
        # include all images within +- 10 min
        for other_date in self.motion_map.keys():
            if date != other_date and abs((date - other_date).total_seconds()) <= 60 * 10:
                filenames = self.motion_map.get(other_date, [])
                imgs += [MotionImage(self, filename, other_date) for filename in filenames]
        return imgs
    
    def generate_motion_image_sets(self) -> list:
        self.__generate_motion_map()
        if len(self.motion_map) == 0:
            raise ValueError("No motion images in session!")
        imgs = []
        dates = sorted(list(self.motion_map.keys()))
        start_date = dates[0]
        for date in dates:
            if abs((date - start_date).total_seconds()) > 60 * 20:
                # end image time series
                yield imgs
                start_date = date
                imgs = []
            # continue time series
            filenames = self.motion_map.get(date, [])
            imgs += [MotionImage(self, filename, date) for filename in filenames]
        # end of all time series
        yield imgs

    def generate_motion_images(self):
        """Yields all motion images in this session.

        Yields:
            MotionImage: A MotionImage
        """
        for file, date in self.motion_dates.items():
            yield MotionImage(self, file, date)

    def generate_lapse_images(self):
        """Yields all lapse images in this session.

        Yields:
            LapseImage: A LapseImage
        """
        for file, date in self.lapse_dates.items():
            yield LapseImage(self, file, date)

    
    def get_closest_lapse_images(self, motion_file: str):
        """Returns the lapse images taken closest before and after this image, respectively.
        If no such image is found, the corresponding returned image will be None.

        Args:
            motion_file (str): Filename of the motion image

        Returns:
            (MotionImage or None, MotionImage or None): Closest lapse images. Each image can be None if not found.
        """
        date: datetime = self.motion_dates[motion_file]
        previous_date = date.replace(minute=0, second=0)
        next_date = previous_date + timedelta(hours=1)
        i = 0
        while not previous_date in self.lapse_map:
            previous_date -= timedelta(hours=1)
            i += 1
            if i > 24:
                # no previous lapse image exists
                previous_date = None
                break
        i = 0
        while not next_date in self.lapse_map:
            next_date += timedelta(hours=1)
            i += 1
            if i > 24:
                # no next lapse image exists
                next_date = None
                break
        if previous_date is not None and len(self.lapse_map[previous_date]) > 1:
            warn(f"There are multiple lapse images for date {previous_date}! Choosing the first one.")
        if next_date is not None and len(self.lapse_map[next_date]) > 1:
            warn(f"There are multiple lapse images for date {next_date}! Choosing the first one.")
        
        previous_img = None if previous_date is None else LapseImage(self, self.lapse_map[previous_date][0], previous_date)
        next_img = None if next_date is None else LapseImage(self, self.lapse_map[next_date][0], next_date)
        return previous_img, next_img

class SessionImage:
    def __init__(self, session: Session, subfolder: str, filename: str, date: datetime):
        self.session = session
        self.subfolder = subfolder
        self.filename = filename
        self.date = date
        if not os.path.isfile(self.get_full_path()):
            raise ValueError(f"File {subfolder}/{filename} in session folder {session.folder} not found!")
    
    def get_full_path(self) -> str:
        return os.path.join(self.session.folder, self.subfolder, self.filename)
    
    def open(self):
        full_path = self.get_full_path()
        print(f"Opening {full_path}...")
        subprocess.call(("xdg-open", full_path))

    def read(self, truncate_y = (40, 40), scale=1, gray=True):
        full_path = self.get_full_path()
        img = io.imread(full_path, as_gray=gray)
        # truncate
        if truncate_y is not None:
            if truncate_y[0] > 0 and truncate_y[1] > 0:
                img = img[truncate_y[0]:(-truncate_y[1]),:]
            elif truncate_y[0] > 0:
                img = img[truncate_y[0]:,:]
            elif truncate_y[1] > 0:
                img = img[:(-truncate_y[1]),:]
        # scale
        if scale is not None and scale < 1:
            img = transform.rescale(img, scale, multichannel=not gray)
        return img
    
    def read_opencv(self, truncate_y = (40, 40), scale=1, gray=True):
        full_path = self.get_full_path()
        img = cv.imread(full_path)
        # grayscale
        if gray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # truncate
        if truncate_y is not None:
            if truncate_y[0] > 0 and truncate_y[1] > 0:
                img = img[truncate_y[0]:(-truncate_y[1])]
            elif truncate_y[0] > 0:
                img = img[truncate_y[0]:]
            elif truncate_y[1] > 0:
                img = img[:(-truncate_y[1])]
        # scale
        if scale is not None and scale < 1:
            img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        return img
        

    def is_daytime(self):
        return 6 <= self.date.hour <= 18
    
    def is_nighttime(self):
        return not self.is_daytime()
    
    def to_ipython_image(self, width=500, height=None):
        return display.Image(filename=self.get_full_path(), width=width, height=height)

class MotionImage(SessionImage):
    def __init__(self, session: Session, filename: str, date: datetime):
        super().__init__(session, "Motion", filename, date)
        if not self.filename in session.motion_dates:
            raise ValueError(f"File name {filename} not in session!")

    def get_closest_lapse_images(self):
        before, after = self.session.get_closest_lapse_images(self.filename)
        rel = -1
        # rel = 0 if motion image was taken at before lapse image, rel = 1 if motion image was taken at after lapse image
        if before is None and after is not None:
            rel = 1
        elif before is not None and after is None:
            rel = 0
        elif before is not None and after is not None:
            rel = (self.date - before.date).total_seconds() / (after.date - before.date).total_seconds()
        else:
            warn("No before and no after image!")
        return before, after, rel
        
class LapseImage(SessionImage):
    def __init__(self, session: Session, filename: str, date: datetime):
        super().__init__(session, "Lapse", filename, date)
        if not self.filename in session.lapse_dates:
            raise ValueError(f"File name {filename} not in session!")
        
