
from typing import Callable, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from physioex.data.constant import get_data_folder
from physioex.data.utils import read_config


def transform_to_sequence(x, y, sequence_length):

    # x shape num_samples x num_features convert it into num_samples x sequence_length x num_features
    # y shape num_samples convert it into num_samples x sequence_length

    x_seq = np.zeros((len(x) - sequence_length + 1, sequence_length, *x.shape[1:]))
    y_seq = np.zeros((len(y) - sequence_length + 1, sequence_length))

    for i in range(len(x) - sequence_length + 1):
        x_seq[i] = x[i : i + sequence_length]
        y_seq[i] = y[i : i + sequence_length]

    return x_seq, y_seq


def create_subject_index_map(df, sequence_length):
    max_windows = (df["num_samples"] - sequence_length + 1).sum()
    window_to_subject = np.zeros(max_windows, dtype=np.int16)
    subject_to_start = np.zeros(df["subject_id"].max() + 1, dtype=np.int32)

    start_index = 0

    for _, row in df.iterrows():
        subject = row["subject_id"]
        num_windows = row["num_samples"] - sequence_length + 1

        window_to_subject[start_index : start_index + num_windows] = subject
        subject_to_start[subject] = start_index

        start_index += num_windows

    return window_to_subject, subject_to_start


def find_subject_for_window(index, window_to_subject, subject_to_start):
    subject = window_to_subject[index]
    start_index = subject_to_start[subject]
    return subject, int(index - start_index)


class PhysioExDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        version: str,
        picks: List[str],  # available [ "Fpz-Cz", "EOG", "EMG" ]
        preprocessing,  # available [ "raw", "xsleepnet" ]
        config_path: str = None,
        sequence_length: int = 21,
        target_transform: Callable = None,
    ):
        self.config = read_config(config_path)

        self.subjects = self.config["subjects_v" + version]

        self.table = pd.read_csv(get_data_folder() + self.config["table"])

        # drop from the table the rows with subject_id not in self.subjects
        self.table = self.table[self.table["subject_id"].isin(self.subjects)]

        self.window_to_subject, self.subject_to_start = create_subject_index_map(
            self.table, sequence_length
        )

        self.split_path = get_data_folder() + self.config["splits_v" + version]
        self.data_path = get_data_folder() + self.config[preprocessing + "_path"]

        self.picks = picks
        self.version = version
        self.preprocessing = preprocessing

        self.mean = None
        self.std = None

        self.L = sequence_length
        self.target_transform = target_transform

        self.input_shape = self.config["shape_" + preprocessing]

    def __len__(self):
        return int(np.sum(self.table["num_samples"] - self.L + 1))

    def __getitem__(self, idx):
        subject_id, relative_id = find_subject_for_window(
            idx, self.window_to_subject, self.subject_to_start
        )

        subject_num_samples = self.table[self.table["subject_id"] == subject_id][
            "num_samples"
        ].values[0]

        input = []
        for pick in self.picks:
            path = self.data_path + f"/{pick}_{subject_id}.dat"

            fp = np.memmap(
                path,
                dtype="float32",
                mode="r",
                shape=(subject_num_samples, *self.input_shape),
            )[relative_id : relative_id + self.L]

            fp = np.expand_dims(fp, axis=1)
            input.append(fp)

        input = np.concatenate(input, axis=1)

        # if len(self.picks) == 1:
        #    input = np.expand_dims(input, axis=0)

        # read the label in the same way

        y = np.memmap(
            self.data_path + f"/y_{subject_id}.dat",
            dtype="int16",
            mode="r",
            shape=(subject_num_samples),
        )[relative_id : relative_id + self.L]

        return torch.tensor(input).float(), torch.tensor(y).view(-1).long()

    def get_num_folds(self):
        pass

    def split(self):
        pass

    def get_sets(self):
        # return the indexes in the table of the train, valid and test subjects
        train_idx = []
        valid_idx = []
        test_idx = []

        for _, row in self.table.iterrows():
            subject_id = int(row["subject_id"])

            start_index = self.subject_to_start[subject_id]
            num_windows = row["num_samples"] - self.L + 1
            indices = np.arange(
                start=start_index, stop=start_index + num_windows
            ).astype(np.int32)

            if row["split"] == 0:
                train_idx.append(indices)
            elif row["split"] == 1:
                valid_idx.append(indices)
            elif row["split"] == 2:
                test_idx.append(indices)

        train_idx = np.concatenate(train_idx)
        valid_idx = np.concatenate(valid_idx)
        test_idx = np.concatenate(test_idx)

        return train_idx, valid_idx, test_idx


class TimeDistributedModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: PhysioExDataset,
        batch_size: int = 32,
        fold: int = 0,
        num_workers: int = 32,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        self.dataset.split(fold)

        self.train_idx, self.valid_idx, self.test_idx = self.dataset.get_sets()

        self.num_workers = num_workers

    def setup(self, stage: str):
        return

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.train_idx),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.valid_idx),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.test_idx),
            num_workers=self.num_workers,
        )


class CombinedTimeDistributedModule(TimeDistributedModule):
    def __init__(
        self,
        dataset: PhysioExDataset,
        batch_size: int = 32,
        num_workers: int = 32,
    ):
        super().__init__(dataset, batch_size, 0, num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
