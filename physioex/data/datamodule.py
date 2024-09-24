import os
from typing import Callable, List, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, SubsetRandomSampler, DistributedSampler, Subset
from physioex.data.dataset import PhysioExDataset


class PhysioExDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling physiological data from multiple datasets.

    Attributes:
        datasets_id (List[str]): List of dataset names.
        num_workers (int): Number of workers for data loading.
        dataset (PhysioExDataset): The dataset object.
        batch_size (int): Batch size for the DataLoader.
        hpc (bool): Flag indicating whether to use high-performance computing.
        train_dataset (Union[PhysioExDataset, Subset]): Training dataset.
        valid_dataset (Union[PhysioExDataset, Subset]): Validation dataset.
        test_dataset (Union[PhysioExDataset, Subset]): Test dataset.
        train_sampler (Union[SubsetRandomSampler, Subset]): Sampler for the training dataset.
        valid_sampler (Union[SubsetRandomSampler, Subset]): Sampler for the validation dataset.
        test_sampler (Union[SubsetRandomSampler, Subset]): Sampler for the test dataset.

    Methods:
        setup(stage: str): Sets up the datasets for different stages.
        train_dataloader(): Returns the DataLoader for the training dataset.
        val_dataloader(): Returns the DataLoader for the validation dataset.
        test_dataloader(): Returns the DataLoader for the test dataset.
    """
    def __init__(
        self,
        datasets: List[str],
        batch_size: int = 32,
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        folds: Union[int, List[int]] = -1,
        data_folder: str = None,
        num_nodes : int = 1,
        num_workers : int = os.cpu_count(),
    ):
        """
        Initializes the PhysioExDataModule.

        Args:
            datasets (List[str]): List of dataset names.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
            preprocessing (str, optional): Type of preprocessing to apply. Defaults to "raw".
            selected_channels (List[int], optional): List of selected channels. Defaults to ["EEG"].
            sequence_length (int, optional): Length of the sequence. Defaults to 21.
            target_transform (Callable, optional): Optional transform to be applied to the target. Defaults to None.
            folds (Union[int, List[int]], optional): Fold number(s) for splitting the data. Defaults to -1.
            data_folder (str, optional): Path to the folder containing the data. Defaults to None.
            num_nodes (int, optional): Number of nodes for distributed training. Defaults to 1.
            num_workers (int, optional): Number of workers for data loading. Defaults to os.cpu_count().
        """
        super().__init__()

        self.datasets_id = datasets
        self.num_workers = num_workers

        self.dataset = PhysioExDataset(
            datasets=datasets,
            preprocessing=preprocessing,
            selected_channels=selected_channels,
            sequence_length=sequence_length,
            target_transform=target_transform,
            data_folder=data_folder,
        )

        self.batch_size = batch_size
        self.hpc = ( num_nodes > 1 )
        
        if isinstance(folds, int):
            self.dataset.split(folds)
        else:
            assert len(folds) == len(
                datasets
            ), "ERR: folds and datasets should have the same length"
            for i, fold in enumerate(folds):
                self.dataset.split(fold, i)

        train_idx, valid_idx, test_idx = self.dataset.get_sets()

        if not self.hpc:
            self.train_dataset = self.dataset
            self.valid_dataset = self.dataset
            self.test_dataset = self.dataset
            
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.valid_sampler = SubsetRandomSampler(valid_idx)
            self.test_sampler = SubsetRandomSampler(test_idx)
        else:
            self.train_dataset = Subset(self.dataset, train_idx)
            self.valid_dataset = Subset(self.dataset, valid_idx)
            self.test_dataset = Subset(self.dataset, test_idx)
            
            self.train_sampler = self.train_dataset
            self.valid_sampler = self.valid_dataset
            self.test_sampler = self.test_dataset

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """        
        return  DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.train_sampler) if self.hpc else self.train_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.valid_sampler) if self.hpc else self.valid_sampler,
            num_workers=self.num_workers,
        )
    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.test_sampler) if self.hpc else self.test_sampler,
            num_workers=self.num_workers,
        )
