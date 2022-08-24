from turtle import pd
from warnings import warn

import numpy as np
import pandas as pd

# helper for accumulating, saving, loading, and displaying dataset statistics
class DatasetStatistics:

    def __init__(self, stats_dict: dict = None, load_from_file: str = None):
        """Create a new statistics instance. The statistics can either be defined
        as a dict (see Dataset#create_statistics()) or loaded from a file.

        Args:
            stats_dict (dict, optional): Dict with statistics. Defaults to None.
            load_from_file (str, optional): File saved with DatasetStatistics#save(). Defaults to None.

        Raises:
            ValueError: If neither stats_dict nor load_from_file is set.
        """
        self.stats = {}
        if stats_dict is not None:
            self.stats = stats_dict
        elif load_from_file is not None:
            self.load(load_from_file)
        else:
            raise ValueError("Please provide 'stats_dict' or 'load_from_file'.")
        self.__update_dataframe()

    def __update_dataframe(self):
        self.df = pd.DataFrame.from_dict(self.stats).transpose()

    def add_total_row(self, row_name = "Z_Total") -> "DatasetStatistics":
        """Add a row to the pandas dataframe with totals of all columns. Should only be called once.

        Args:
            row_name (str, optional): Name of the new row. Defaults to "Z_Total".

        Returns:
            DatasetStatistics: self
        """
        if row_name in self.stats:
            warn(f"{row_name} is already a defined row")
            return self
        self.stats[row_name] = {}
        # iterate over all folders and subfolders
        for folder in self.stats:
            if folder != row_name:
                for subfolder in self.stats[folder]:
                    # add to total row
                    if subfolder in self.stats[row_name]:
                        self.stats[row_name][subfolder] += self.stats[folder][subfolder]
                    else:
                        self.stats[row_name][subfolder] = self.stats[folder][subfolder]
        self.__update_dataframe()
        return self
    
    def save(self, filename = "dataset_stats.npy"):
        """Save statistics stored in this instance to a file using numpy.

        Args:
            filename (str, optional): Target file name. Defaults to "dataset_stats.npy".
        """
        np.save(filename, self.stats)
        print(f"Saved to {filename}.")
    
    def load(self, filename = "dataset_stats.npy"):
        """Load statistics from a file using numpy.

        Args:
            filename (str, optional): Target file name. Defaults to "dataset_stats.npy".
        """
        self.stats = np.load(filename, allow_pickle=True).tolist()
        self.__update_dataframe()
        print(f"Loaded from {filename}.")

    def view(self, col_order = ["Lapse", "Motion", "Full", "Total"]) -> pd.DataFrame:
        """Display the statistics dataframe.

        Args:
            col_order (list, optional): Order of columns. Defaults to ["Lapse", "Motion", "Full", "Total"].

        Returns:
            pd.DataFrame: data frame
        """
        return self.df.sort_index()[col_order]
    
    def plot_sessions(self, cols = ["Lapse", "Motion", "Full"], figsize = (20, 10), style = {"width": 2}, exclude_last_row = False):
        """Plot the statistics dataframe as a bar plot.

        Args:
            cols (list, optional): Columns to include. Defaults to ["Lapse", "Motion", "Full"].
            figsize (tuple, optional): Plot size. Defaults to (20, 10).
            style (dict, optional): Additional style arguments. Defaults to {"width": 2}.
            exclude_last_row (bool, optional): If True, the last row will not be plotted. Defaults to False.

        Returns:
            _type_: _description_
        """
        df = self.df[cols]
        # Plot lapse, motion, full columns without the last row (Z_Total)
        if exclude_last_row:
            df = df.iloc[:-1]
        return df.plot.bar(figsize=figsize, style=style)