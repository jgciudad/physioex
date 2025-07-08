import os
from pathlib import Path
from typing import List, Union
import time

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
from umap import UMAP

from physioex.data import PhysioExDataModule
from physioex.train.models.load import load_model
from physioex.train.networks.base import SleepModule
from physioex.train.bin.parser import PhysioExParser


def visualize(
    datasets: Union[List[str], str, PhysioExDataModule],
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
        
    embeddings_concat = None
    y_concat = None
    for _, test_datamodule in enumerate(datamodule):
        dataloader = test_datamodule.test_dataloader(shuffle=True)
        d_iter = iter(dataloader)
        
        # for batch in tqdm(dataloader, desc="Computing embeddings"):
        print('n_batches:', int(0.1*len(d_iter)))
        for _ in tqdm(range(int(0.1*len(d_iter))), desc="Computing embeddings"):
            x, y = next(d_iter)
            
            embeddings = model.visualization_encoding(x.cuda())

            if embeddings_concat is None:
                embeddings_concat = embeddings.cpu().detach().numpy()
                y_concat = y.flatten().cpu().detach().numpy()
            else:
                embeddings_concat = np.concatenate((embeddings_concat, embeddings.cpu().detach().numpy()), axis=0)
                y_concat = np.concatenate((y_concat, y.flatten().cpu().detach().numpy()), axis=0)
                    
        if len(np.unique(y_concat)) == 5:
            label_map = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
            y_concat = np.vectorize(label_map.get)(y_concat)
        elif len(np.unique(y_concat)) == 3:
            label_map = {0: "W", 1: "N", 2: "R"}
            y_concat = np.vectorize(label_map.get)(y_concat)
            
        # # Select a subsample of the embeddings and labels randomly
        # num_samples = int(0.01 * len(embeddings_concat))
        # indices = np.random.choice(len(embeddings_concat), num_samples, replace=False)
        # embeddings_concat = embeddings_concat[indices]
        # y_concat = y_concat[indices]
        
        print('Computing UMAP')
        initial_time = time.time()
        umap = UMAP(n_components=2, random_state=42)
        embeddings_umap = umap.fit_transform(embeddings_concat)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=embeddings_umap[:, 0], y=embeddings_umap[:, 1], hue=y_concat, palette=sns.color_palette("bright", len(np.unique(y_concat))))
        plt.title("UMAP of Embeddings")
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.legend(title="Classes")
        save_path = Path(results_path) / 'embeddings' / 'umap'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'{test_datamodule.datasets_id[0]}.png')
        print(f'UMAP took {time.time() - initial_time} seconds')
        
        print('Computing PCA')
        initial_time = time.time()
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings_concat)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], hue=y_concat, palette=sns.color_palette("bright", len(np.unique(y_concat))))
        plt.title("PCA of Embeddings")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Classes")
        save_path = Path(results_path) / 'embeddings' / 'pca'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'{test_datamodule.datasets_id[0]}.png')
        print(f'PCA took {time.time() - initial_time} seconds')
        
        print('Computing t-SNE')
        initial_time = time.time()
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings_concat)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], hue=y_concat, palette=sns.color_palette("bright", len(np.unique(y_concat))))
        plt.title("t-SNE of Embeddings")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(title="Classes")
        save_path = Path(results_path) / 'embeddings' / 'tsne'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'{test_datamodule.datasets_id[0]}.png')
        print(f't-SNE took {time.time() - initial_time} seconds')

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