# COMBINED PROTOTYPES


import os
from pathlib import Path
from typing import List, Union
import time

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pytorch_lightning import Trainer
from torch import set_float32_matmul_precision
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sympy import divisors

from physioex.data import PhysioExDataModule
from physioex.train.models.load import load_model
from physioex.train.networks.base import SleepModule
from physioex.train.bin.parser import PhysioExParser
from physioex.preprocess.utils.signal import OnlineVariance

def lowest_divisor_pair(n):
    for a in range(2, int(n**0.5) + 1):
        if n % a == 0:
            b = n // a
            return (a, b)
    return (1, n) 

def get_groups(datamodule: PhysioExDataModule, 
               dataset_idx: torch.Tensor,
               subject_id: torch.Tensor) -> List[str]:
    
    groups = []
    
    for d_idx, s_id in zip(dataset_idx, subject_id):
        d_idx = int(d_idx)
        s_id = int(s_id)
        
        d_table = datamodule.dataset.readers[d_idx].reader.table
        
        if 'group' in d_table.columns:
            group = d_table[d_table['subject_id'] == s_id]['group'].values[0]
        else:
            group = 'Unique group'
        
        groups.append(group)
    
    return groups
    
def subject_density(prototype_predictions: np.ndarray[np.int_],
                    channels: List[str],
                    n_proto: int,
                    subjects: np.ndarray[np.int_],
                    results_path: str):
    
    # Plot the ratio of sleep stage per prototype. One plot per channel. 
    df_dict = {'subjects': subjects}
    for c_idx, c in enumerate(channels): 
        df_dict[c] = prototype_predictions[:,c_idx]
    df = pd.DataFrame(df_dict)
    
    # Create a barplot with value counts of df['subjects']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['subjects'].value_counts().index, y=df['subjects'].value_counts(normalize=True).values, palette="viridis")
    plt.title("Value Counts of Subjects")
    plt.xlabel("Subjects")
    plt.ylabel("Count")
    plt.tight_layout()
    save_path = Path(results_path) / 'prototype_density_combined' / 'subject_density'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'subject_counts.png'))
    plt.close()
    
    for c_idx, c in enumerate(channels):
        ratio_table = df.groupby(c)['subjects'].value_counts(normalize=True).unstack().fillna(0)
        n_rows, n_cols = lowest_divisor_pair(n_proto[c_idx])

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25), dpi=300)
        if axes.ndim == 1:
            axes = np.expand_dims(axes, axis=1)
            
        for i in range(axes.size):
            ax = axes[int(i // axes.shape[1]), int(i % axes.shape[1])]
            if i not in ratio_table.index:
                ax.axis('off')
            else:
                ax.pie(ratio_table.loc[i], labels=ratio_table.loc[i].index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 25})
            ax.set_title(f'Prototype {i} - Count {(df[c] == i).sum()}', fontsize=25)
        plt.tight_layout()
        save_path = Path(results_path) / 'prototype_density_combined' / 'subject_density'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{c}.png'))
        
        ratio_table.plot(kind='bar', stacked=True)
        plt.savefig(os.path.join(save_path, f'{c}2.png'))

def prototype_density(prototype_predictions: np.ndarray[np.int_],
                      y: np.ndarray[np.int_],
                      channels: List[str],
                      results_path: str) -> None:
    
    # Plot the ratio of sleep stage per prototype. One plot per channel. 
    df_dict = {'y': y}
    for c_idx, c in enumerate(channels): 
        df_dict[c] = prototype_predictions[:,c_idx]
    df = pd.DataFrame(df_dict)
    
    ratio_table = df.groupby(channels)['y'].value_counts(normalize=True).unstack().fillna(0)
    n_proto = ratio_table.shape[0]
    
    n_rows, n_cols = lowest_divisor_pair(n_proto)
            
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 9*n_rows), dpi=300, tight_layout=True)
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for i in range(axes.size):
        ax = axes[int(i // axes.shape[1]), int(i % axes.shape[1])]
        if i > len(ratio_table.index) - 1:
                ax.axis('off')
        else:
            ax.pie(ratio_table.iloc[i].sort_index(), labels=ratio_table.iloc[i].sort_index().index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 25}, colors=sns.color_palette("Set1"))
        
            prototype_names= [c + str(ratio_table.index[i][c_idx]) for c_idx, c in enumerate(channels)]
            prototype_names = " ".join(prototype_names)
            subplot_title = "P " + prototype_names
            
            counts = df[channels].value_counts()[ratio_table.index[i]]
            ax.set_title(subplot_title + f' {(counts / len(y) * 100):.2f}%', fontsize=25)
    save_path = Path(results_path) / 'prototype_density_combined' / 'specificity'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'spec.png'))
    ratio_table.to_csv(os.path.join(save_path, 'spec.csv'))
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 7), dpi=300)
    
    # Group by y, EEG, EMG and count occurrences
    group_by_columns = ['y'] + channels
    group_counts = df.groupby(group_by_columns).size().reset_index(name='count')

    # Total count for each y
    total_counts = df['y'].value_counts().to_dict()

    # Add ratio column
    group_counts['ratio'] = group_counts.apply(lambda row: row['count'] / total_counts[row['y']], axis=1)
    group_counts['combined_prototype'] = group_counts.apply(lambda row: '_'.join([c + str(row[c]) for c in channels]), axis=1)
    group_counts = group_counts.sort_values(by=channels[0])

    sns.barplot(data=group_counts, x='combined_prototype', y='ratio', hue='y', ax=ax, palette=sns.color_palette("Set1"), hue_order=sorted(group_counts['y'].dropna().unique()), order=sorted(group_counts['combined_prototype'].unique()))
    ax.set_xlabel('Combined prototype')
    ax.legend(title='Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    save_path = Path(results_path) / 'prototype_density_combined' / 'sensitivity'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sensitivity.png'))
    group_counts.to_csv(os.path.join(save_path, 'sensitivity.csv'))
    
    # Plot raw counts
    group_counts = df.groupby(channels).size().reset_index(name='count')
    group_counts['combined_prototype'] = group_counts.apply(lambda row: '_'.join([c + str(row[c]) for c in channels]), axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(15, 7), dpi=300)
    sns.barplot(data=group_counts, x='combined_prototype', y='count', ax=ax, order=group_counts['combined_prototype'].unique())
    plt.title('Value Counts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = Path(results_path) / 'prototype_density_combined'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'counts.png'))
    plt.close()



            
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

    # seed_everything(42, workers=True)
    set_float32_matmul_precision("medium")

    datamodule_kwargs["batch_size"] = batch_size
    # datamodule_kwargs["hpc"] = hpc
    datamodule_kwargs["folds"] = fold
    datamodule_kwargs["num_nodes"] = num_nodes
    # datamodule_kwargs["evaluate_on_whole_night"] = True

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
    
    
    learned_prototypes = [chan_codebook.codebook.cpu().detach().numpy() for chan_codebook in model.nn.prototype]
    n_proto = [chan_codebook.shape[0] for chan_codebook in learned_prototypes]
    channels = datamodule_kwargs["selected_channels"]
    n_channels = len(channels)
    assert len(n_proto) == n_channels, "Number of prototypes must match number of channels"
    
    # Create OnlineVariance objects for each prototype and channel
    mean_psd = {}
    for c_idx, c in enumerate(channels):
        mean_psd[c] = {}
        for i in range(n_proto[c_idx]):
            mean_psd[c][i] = OnlineVariance((129,))
    
    # Initialize a dictionary to store embeddings for each channel
    embeddings = {}
    for c in channels:
        embeddings[c] = []
        
    # Initialize variables to store y and prototype predictions
    y = []
    proto_idx = []
    subjects_id = []
    # transition_matrix = np.zeros((n_channels, n_proto, n_proto), dtype=np.int_) # TODO 
    
    for _, test_datamodule in enumerate(datamodule):
        dataloader = test_datamodule.test_dataloader(shuffle=True)
        # dataloader = test_datamodule.train_dataloader()
        d_iter = iter(dataloader)
        
        samples = 70000
        n_batches = int(samples / batch_size)
        for _ in tqdm(range(n_batches), desc="Computing embeddings"):
        # for batch in dataloader:
            
            try:
                x_, y_, subject_id_, _ = next(d_iter)
            except StopIteration:
                print("End of dataloader")
                break
            
            batch_size, L, nchan, T, F = x_.shape
            sequence_center = L // 2 + 1 # index of the center element of the sequence
            
            embeddings_, _, proto_idx_, _, alphas = model.nn.get_prototypes(x_.cuda())
            
            proto_idx_ = proto_idx_.cpu().numpy()
            # transition_matrix = count_transitions(proto_idx_, transition_matrix) # TODO
            
            x_ = x_ * test_datamodule.dataset.readers[0].reader.std + test_datamodule.dataset.readers[0].reader.mean

            embeddings_ = embeddings_[:, sequence_center, :, :, :].detach().cpu().numpy() # take center element of the sequence
            proto_idx_ = proto_idx_[:, sequence_center,:, 0]
            alphas = alphas[:, sequence_center, :, 0, :].detach().cpu().numpy()
            x_ = x_[:, sequence_center].cpu().numpy()
            y_ = y_[:, sequence_center].cpu().numpy()
            
            # select time window that was sampled
            sampled_x = np.einsum('bctf, bct -> bcf', x_, alphas)
                        
            # store psd for each prototype and channel
            for c_idx, c in enumerate(channels):
                for p in np.unique(proto_idx_[:, c_idx]):
                    p_filter_idx = np.where(proto_idx_[:, c_idx] == p)[0]
                    
                    if len(p_filter_idx) > 0:
                        signal = sampled_x[p_filter_idx, c_idx]
                        mean_psd[c][p].add(signal)

            # store embeddings for each channel
            for c_idx, c in enumerate(channels):
                embeddings[c].append(embeddings_[:,c_idx])

            # store y and prototype predictions
            y.append(y_)
            subjects_id.append(subject_id_)
            proto_idx.append(proto_idx_)
        
        # concatenate all batches
        for c_idx, c in enumerate(channels):
            embeddings[c] = np.concatenate(embeddings[c], axis=0)
        y = np.concatenate(y, axis=0)
        subjects_id = np.concatenate(subjects_id, axis=0)
        proto_idx = np.concatenate(proto_idx, axis=0)
                    
        if len(np.unique(y)) == 5:
            label_map = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
            y = np.vectorize(label_map.get)(y)
        elif len(np.unique(y)) == 3:
            label_map = {0: "W", 1: "N", 2: "R"}
            y = np.vectorize(label_map.get)(y)
        
        # plot stage ratio in each prototype and prototype counts
        prototype_density(proto_idx, y, channels, results_path)
        
        # plot subject density in each prototype
        # subject_density(proto_idx, channels, n_proto, subjects_id, results_path) # TODO



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
        fold=parser["fold"],
        model_class=parser["model"],
        model_config=parser["model_kwargs"],
        batch_size=parser["batch_size"],
        hpc=parser["hpc"],
        num_nodes=parser["num_nodes"],
        checkpoint_path=parser["checkpoint_path"],
        results_path=parser["results_path"],
        aggregate_datasets=parser["aggregate"],
    )