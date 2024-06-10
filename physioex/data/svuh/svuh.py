from typing import Callable, List

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from physioex.data.base import PhysioExDataset, create_subject_index_map
from physioex.data.constant import get_data_folder


class Svuh(PhysioExDataset):
    def __init__(
        self,
        version: str = None,
        picks: List[str] = ["EEG"],  # available [ "EEG", "EOG", "EMG" ]
        preprocessing: str = "raw",  # available [ "raw", "xsleepnet" ]
        sequence_length: int = 21,
        target_transform: Callable = None,
    ):
        assert preprocessing in [
            "raw",
            "xsleepnet",
        ], "preprocessing should be one of 'raw'-'xsleepnet'"
        for pick in picks:
            assert pick in [
                "EEG",
                "EOG",
                "EMG",
                "ECG"
            ], "pick should be one of 'EEG, 'EOG', 'EMG', 'ECG'"

        self.table = pd.read_csv(get_data_folder() + "/svuh/table.csv")

        self.subjects = self.table["subject_id"].values.astype(np.int16)

        self.window_to_subject, self.subject_to_start = create_subject_index_map(
            self.table, sequence_length
        )

        self.split_path = get_data_folder() + f"/svuh/splitting.npz"

        self.data_path = get_data_folder() + f"/svuh/{preprocessing}/"

        self.picks = picks
        self.version = version
        self.preprocessing = preprocessing

        self.mean = None
        self.std = None

        self.L = sequence_length
        self.target_transform = target_transform

        if self.preprocessing == "raw":
            self.input_shape = [3000]
        else:
            self.input_shape = [29, 129]

        scaling_file = np.load(get_data_folder() + f"/svuh/{preprocessing}/scaling.npz")

        EEG_mean, EOG_mean, EMG_mean, ECG_mean = scaling_file["mean"]
        EEG_std, EOG_std, EMG_std, ECG_std = scaling_file["std"]

        self.mean = []
        self.std = []

        if "EEG" in self.picks:
            self.mean.append(EEG_mean)
            self.std.append(EEG_std)
        if "EOG" in self.picks:
            self.mean.append(EOG_mean)
            self.std.append(EOG_std)
        if "EMG" in self.picks:
            self.mean.append(EMG_mean)
            self.std.append(EMG_std)
        if "ECG" in self.picks:
            self.mean.append(ECG_mean)
            self.std.append(ECG_std)

        self.mean = torch.tensor(np.array(self.mean)).float()
        self.std = torch.tensor(np.array(self.std)).float()

    def get_num_folds(self):
        return 1

    def split(self, fold: int = 0):

        split_file = np.load(get_data_folder() + f"/svuh/{self.preprocessing}/scaling.npz")

        train_subjects = split_file["train_subjects"].astype( np.int16 )
        valid_subjects = split_file["valid_subjects"].astype( np.int16 )
        test_subjects = split_file["test_subjects"].astype( np.int16 )
        
        # add a column to the table with 0 if the subject is in train, 1 if in valid, 2 if in test

        split = np.zeros(len(self.table)).astype(np.int8)
        split[self.table["subject_id"].isin(test_subjects)] = 2
        split[self.table["subject_id"].isin(valid_subjects)] = 1

        self.table["split"] = split

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        y = y - 1

        x = (x - self.mean) / self.std

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y
