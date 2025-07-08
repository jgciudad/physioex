import os
from pathlib import Path
from typing import List, Union

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import set_float32_matmul_precision
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.signal

from physioex.data import PhysioExDataModule
from physioex.train.models.load import load_model
from physioex.train.networks.base import SleepModule
from physioex.train.bin.parser import PhysioExParser
from physioex.preprocess.utils.signal import OnlineVariance

    
def visualize(
    datasets: Union[List[str], str, PhysioExDataModule],
    channel: int = 0,
    datamodule_kwargs: dict = {},
    model: SleepModule = None,  # if passed model_class, model_config and resume are ignored
    model_class=None,
    model_config: dict = None,
    batch_size: int = 128,
    fold: int = -1,
    hpc: bool = False,
    checkpoint_path: str = None,
    results_path: str = None,
    num_nodes: int = 1,
    aggregate_datasets: bool = False,
) -> pd.DataFrame:

    seed_everything(42, workers=True)
    set_float32_matmul_precision("medium")

    datamodule_kwargs["batch_size"] = batch_size
    # datamodule_kwargs["hpc"] = hpc
    datamodule_kwargs["folds"] = fold
    datamodule_kwargs["num_nodes"] = num_nodes

    ##### DataModule Setup #####
    if isinstance(datasets, PhysioExDataModule):
        datamodule = [datasets]
    elif isinstance(datasets, str):
        datamodule = [
            PhysioExDataModule(
                datasets=[datasets],
                **datamodule_kwargs,
            )
        ]
    elif isinstance(datasets, list):
        if aggregate_datasets:
            datamodule = PhysioExDataModule(
                datasets=datasets,
                **datamodule_kwargs,
            )
        else:
            datamodule = []
            for dataset in datasets:
                datamodule.append(
                    PhysioExDataModule(
                        datasets=[dataset],
                        **datamodule_kwargs,
                    )
                )
    else:
        raise ValueError("datasets must be a list, a string or a PhysioExDataModule")

    
    ##### Raw DataModule Setup #####
    datamodule_kwargs["preprocessing"] = 'raw'
    if isinstance(datasets, PhysioExDataModule):
        raw_datamodule = [datasets]
    elif isinstance(datasets, str):
        raw_datamodule = [
            PhysioExDataModule(
                datasets=[datasets],
                **datamodule_kwargs,
            )
        ]
    elif isinstance(datasets, list):
        if aggregate_datasets:
            raw_datamodule = PhysioExDataModule(
                datasets=datasets,
                **datamodule_kwargs,
            )
        else:
            raw_datamodule = []
            for dataset in datasets:
                raw_datamodule.append(
                    PhysioExDataModule(
                        datasets=[dataset],
                        **datamodule_kwargs,
                    )
                )
    else:
        raise ValueError("datasets must be a list, a string or a PhysioExDataModule")

    ########### Resuming Model if needed else instantiate it ############:
    if model is None:
        model = load_model(
            model=model_class,
            model_kwargs=model_config,
            ckpt_path=checkpoint_path,
        )

    ########### Trainer Setup ############
    from lightning.pytorch.accelerators import find_usable_cuda_devices

    devices = find_usable_cuda_devices(-1)

    trainer = Trainer(
        devices=devices,
        strategy="ddp" if (num_nodes > 1 or len(devices) > 1) else "auto",
        num_nodes=num_nodes,
        # callbacks=[progress_bar_callback],
        deterministic=True,
    )
    
    if results_path is not None:
        Path(results_path).mkdir(parents=True, exist_ok=True)
        
    for t_idx, test_datamodule in enumerate(datamodule):
        dataloader = test_datamodule.test_dataloader()
        d_iter = iter(dataloader)
        
        raw_dataloader = raw_datamodule[t_idx].test_dataloader()
        raw_d_iter = iter(raw_dataloader)

        mean_psd = {}
        for i in range(model.n_classes):
            mean_psd[i] = OnlineVariance((128,))
        
        print('n_batches:', int(0.3*len(d_iter)))
        for i in tqdm(range(len(d_iter)), desc="Computing embeddings"):
        # for i in tqdm(range(1000), desc="Computing embeddings"):
            x, y = next(d_iter)
            raw_x, _ = next(raw_d_iter)
            
            if np.random.rand() < 0.3: # compute only 20% of the dataset
                raw_x = raw_x[:, :, channel]
                raw_x = raw_x.reshape(-1, raw_x.shape[-1])
            
                y_hat = model(x.cuda())
                y_hat = y_hat.cpu()
                y_hat = torch.argmax(y_hat.reshape(-1, y_hat.shape[-1]), dim=1)
                            
                freqs, psd_x = scipy.signal.welch(raw_x.detach().cpu().numpy(), fs=100, nperseg=256)
                psd_x = psd_x[:,1:]
                freqs = freqs[1:]

                # integral = scipy.integrate.simpson(psd_x, freqs, axis=1)
                # psd_x = psd_x / integral[:, None]
                                
                for stage in torch.unique(y_hat):
                    mean_psd[int(stage)].add(psd_x[y_hat == stage])
            
        if len(mean_psd.keys()) == 5:
            label_map = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
        elif len(mean_psd.keys()) == 3:
            label_map = {0: "W", 1: "N", 2: "R"}
            
        fig, ax = plt.subplots(1, len(mean_psd.keys()), figsize=(20, 7.5))
        for s_idx, stage in enumerate(mean_psd.keys()):
            s_mean, s_std = mean_psd[stage].compute()
                        
            ax[s_idx].plot(freqs, s_mean, color=plt.cm.viridis(s_idx / len(mean_psd.keys())))
            ax[s_idx].fill_between(freqs, s_mean - s_std, s_mean + s_std, color=plt.cm.viridis(s_idx / len(mean_psd.keys())), alpha=0.5)
            ax[s_idx].set_title(f"Stage {label_map[stage]}")
            ax[s_idx].set_xlabel("Frequency (Hz)")
            ax[s_idx].set_yscale('log')
            ax[s_idx].set_ylabel(r"PSD ($V^2/Hz$)")
        # Unify ylims for all subplots
        ylims = [ax[s_idx].get_ylim() for s_idx in range(len(mean_psd.keys()))]
        min_ylim = min([ylim[0] for ylim in ylims])
        max_ylim = max([ylim[1] for ylim in ylims])
        for s_idx in range(len(mean_psd.keys())):
            ax[s_idx].set_ylim(min_ylim, max_ylim)
        plt.tight_layout()
        save_path = Path(results_path) / 'psd_predictions'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'{test_datamodule.datasets_id[0]}.png')
            

if __name__ == "__main__":
    
    parser = PhysioExParser.test_parser()

    datamodule_kwargs = {
        "selected_channels": parser["selected_channels"],
        "sequence_length": parser["sequence_length"],
        "target_transform": parser["target_transform"],
        "preprocessing": parser["preprocessing"],
        "task": parser["model_task"],
        "data_folder": parser["data_folder"],
        "num_workers": parser["num_workers"],
    }

    visualize(
        datasets=parser["datasets"],
        datamodule_kwargs=datamodule_kwargs,
        model=None,
        model_class=parser["model"],
        model_config=parser["model_kwargs"],
        batch_size=parser["batch_size"],
        hpc=parser["hpc"],
        num_nodes=parser["num_nodes"],
        checkpoint_path=parser["checkpoint_path"],
        results_path=parser["results_path"],
        aggregate_datasets=parser["aggregate"],
    )