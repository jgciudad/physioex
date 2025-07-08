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
from scipy.spatial import Voronoi, voronoi_plot_2d

from physioex.data import PhysioExDataModule
from physioex.train.models.load import load_model
from physioex.train.networks.base import SleepModule
from physioex.train.bin.parser import PhysioExParser
from physioex.preprocess.utils.signal import OnlineVariance

def plot_labels(y,
                groups,
                results_path) -> None:
    df = pd.DataFrame({'y': y, 'groups': groups})
    
    df = df.groupby(['groups']).value_counts(normalize=True).reset_index(name='count')
    
    
    plt.figure(figsize=(16, 8))
    sns.barplot(data=df, x='y', y='count', hue='groups', palette="Set1", order=sorted(df['y'].value_counts().index))
    plt.title("Label Counts per Group")
    plt.xlabel("Labels")
    plt.ylabel("Stage ratio")
    plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.tight_layout()
    save_path = Path(results_path) 
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'label_counts.png'))
    plt.close()
    
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
                    n_proto: List[int],
                    subjects: np.ndarray[np.int_],
                    results_path: str,
                    attribute: str = 'subjects') -> None:
    
    # Plot the ratio of sleep stage per prototype. One plot per channel. 
    df_dict = {attribute: subjects}
    for c_idx, c in enumerate(channels): 
        df_dict[c] = prototype_predictions[:,c_idx]
    df = pd.DataFrame(df_dict)
    
    # Create a barplot with value counts of df['subjects']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df[attribute].value_counts().index, y=df[attribute].value_counts(normalize=True).values, palette="viridis")
    plt.title("Value Counts of Subjects")
    plt.xlabel("Subjects")
    plt.ylabel("Count")
    plt.tight_layout()
    save_path = Path(results_path) / 'prototype_density' / (attribute + '_density')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, attribute + '_counts.png'))
    plt.close()
    
    for c_idx, c in enumerate(channels):
        ratio_table = df.groupby(c)[attribute].value_counts(normalize=True).unstack().fillna(0)
        
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
        save_path = Path(results_path) / 'prototype_density' / (attribute + '_density')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{c}.png'))
        
        ratio_table.plot(kind='bar', stacked=True)
        plt.savefig(os.path.join(save_path, f'{c}_bars.png'))


def count_transitions(prototype_predictions: np.ndarray[np.int_],
                      transition_matrix: np.ndarray[np.int_]):
    
    sequence_center = prototype_predictions.shape[1] // 2 + 1
    prototype_predictions = prototype_predictions[:, sequence_center-1 : sequence_center+1, 0]
    
    for r in prototype_predictions:
        transition_matrix[r[0], r[1]] += 1
    
    return transition_matrix
        
        
def average_psd(channels: List[str],
                mean_psd: dict,
                results_path: str) -> None:
    '''Plot the average psd for each prototype and channel'''
    
    n_channels = len(channels)
    cmap = plt.get_cmap("tab10", n_channels)

    for c_idx, c in enumerate(channels):
        n_proto_c = len(mean_psd[c])
        
        n_rows, n_cols = lowest_divisor_pair(n_proto_c)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), dpi=300, tight_layout=True)
        fig_linear, axes_linear = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), dpi=300, tight_layout=True)
        if axes.ndim == 1:
            axes = np.expand_dims(axes, axis=1)
            axes_linear = np.expand_dims(axes_linear, axis=1)
        
        for i in range(axes.size):
            ax = axes[int(i // axes.shape[1]), int(i % axes.shape[1])]
            ax_linear = axes_linear[int(i // axes.shape[1]), int(i % axes.shape[1])]

            if i < n_proto_c:
                s_mean, s_std = mean_psd[c][i].compute()
                        
                ax.set_xticks(np.linspace(0, s_mean.size - 1, 11))
                ax.set_xticklabels(np.arange(0, 51, 5))
                ax.set_xlim(0, s_mean.size - 1)
                ax.plot(s_mean, color=cmap(c_idx))
                ax.fill_between(np.arange(s_mean.size), s_mean - s_std, s_mean + s_std, alpha=0.5, color=cmap(c_idx))
                ax.set_title(f"P{i} "+ c)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("PSD (dB)")
                
                # Convert s_mean and s_std from dB to linear
                s_mean = 10 ** (s_mean / 20)
                s_std = 10 ** (s_std / 20)
                s_mean = s_mean[:s_mean.size//2]
                s_std = s_std[:s_std.size//2]
                ax_linear.set_xticks(np.linspace(0, s_mean.size - 1, 6))
                ax_linear.set_xticklabels(np.arange(0, 26, 5))
                ax_linear.set_xlim(0, s_mean.size - 1)
                ax_linear.plot(s_mean, color=cmap(c_idx))
                # ax_linear.fill_between(np.arange(s_mean.size), s_mean - s_std, s_mean + s_std, alpha=0.5, color=cmap(c_idx))
                ax_linear.set_title(f"P{i} "+ c)
                ax_linear.set_xlabel("Frequency (Hz)")
                ax_linear.set_ylabel("Power (linear)")
            else:
                ax.axis('off')
                ax_linear.axis('off')
        
        fig.savefig(os.path.join(results_path, f'{c}_psd_dB.png'))
        fig_linear.savefig(os.path.join(results_path, f'{c}_psd_linear.png'))
    
        
def prototype_density(prototype_predictions: np.ndarray[np.int_],
                      n_proto: List[int],
                      y: np.ndarray[np.int_],
                      channels: List[str],
                      results_path: str) -> None:
    # Plot the ratio of sleep stage per prototype. One plot per channel. 
    df_dict = {'y': y}
    for c_idx, c in enumerate(channels): 
        df_dict[c] = prototype_predictions[:,c_idx]
    df = pd.DataFrame(df_dict)
    
    for c_idx, c in enumerate(channels):
        ratio_table = df.groupby(c)['y'].value_counts(normalize=True).unstack().fillna(0)
        
        n_rows, n_cols = lowest_divisor_pair(n_proto[c_idx])
                
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 9*n_rows), dpi=300, tight_layout=True)
        if axes.ndim == 1:
            axes = np.expand_dims(axes, axis=1)
            
        for i in range(axes.size):
            ax = axes[int(i // axes.shape[1]), int(i % axes.shape[1])]
            if i not in ratio_table.index:
                ax.axis('off')
            else:
                ax.pie(ratio_table.loc[i].sort_index(), labels=ratio_table.loc[i].sort_index().index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 25}, colors=sns.color_palette("Set1"))
                ax.set_title(f'Prototype {i}, {((df[c] == i).sum() / len(y) * 100):.2f}%', fontsize=25)
        save_path = Path(results_path) / 'prototype_density' / 'specificity'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{c}.png'))
        ratio_table.to_csv(os.path.join(save_path, f'{c}.csv'))
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(15, 7), dpi=300)
    for c_idx, c in enumerate(channels):
        if isinstance(axes, np.ndarray):
            ax = axes[c_idx]
        else:
            ax = axes
        ratio_table = df.groupby('y')[c].value_counts(normalize=True).unstack().fillna(0).T
        ratio_table = ratio_table.reset_index() 
        melted_df = ratio_table.melt(id_vars=c, var_name='Stage', value_name='Sensitivity')
        for proto in range(n_proto[c_idx]):
            if proto not in melted_df[c].unique():
                melted_df = pd.concat([melted_df, pd.DataFrame({c: [proto], 'Stage': [np.nan], 'Sensitivity': [np.nan]})], ignore_index=True)
        axx = sns.barplot(data=melted_df, x=c, y='Sensitivity', hue='Stage', ax=ax, palette=sns.color_palette("Set1"), hue_order=sorted(melted_df['Stage'].dropna().unique()))

        # Add labels on top of bars
        for p in axx.patches:
            height = p.get_height()
            axx.text(
                p.get_x() + p.get_width() / 2,
                height,
                f'{height:.3f}',  # format as needed
                ha='center',
                va='bottom'
            )
        ax.set_title(c)
        ax.set_xlabel('Prototype')
        ax.legend(title='Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path = Path(results_path) / 'prototype_density' / 'sensitivity'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.xticks(rotation=90)  # Rotate x-axis tick labels
    plt.savefig(os.path.join(save_path, 'sensitivity.png'))
    melted_df.to_csv(os.path.join(save_path, 'sensitivity.csv'))
        
    # Plot the counts of each prototype per channel
    value_counts = {c: df[c].value_counts() for c in channels}
    
    bar_data = []
    for c_idx, c in enumerate(channels):
        for idx in range(n_proto[c_idx]):
            bar_data.append({'Channel': c, 'Prototype': idx, 'Count': value_counts[c].get(idx, 0)})
    bar_df = pd.DataFrame(bar_data)
    
    plt.figure(figsize=(16, 8))
    sns.barplot(data=bar_df, x='Prototype', y='Count', hue='Channel', dodge=True, palette=sns.color_palette("Set1"))
    plt.title('Value Counts for EEG, EOG, and EMG')
    plt.xticks(rotation=45)
    plt.legend(title='Channel')
    plt.tight_layout()
    save_path = Path(results_path) / 'prototype_density'
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
    transition_matrices = {}
    for c_idx, c in enumerate(channels):
        mean_psd[c] = {}
        for i in range(n_proto[c_idx]):
            mean_psd[c][i] = OnlineVariance((129,))
        
        transition_matrices[c] = np.zeros((n_proto[c_idx], n_proto[c_idx]), dtype=np.int_)
    
    # Initialize a dictionary to store embeddings of each channel
    embeddings = {}
    for c in channels:
        embeddings[c] = []
        
    # Initialize variables to store y, prototype predictions, subject ids
    y = []
    proto_idx = []
    subjects_id = []
    groups = []

    
    for _, test_datamodule in enumerate(datamodule):
        dataloader = test_datamodule.test_dataloader(shuffle=True)
        # dataloader = test_datamodule.train_dataloader()
        
        d_iter = iter(dataloader)
        
        samples = 70000 # number of epochs for visualization
        n_batches = int(samples / batch_size)
        for batch in tqdm(range(n_batches), desc="Getting prototypes"):
            
            # Get the next batch                
            try:
                x_, y_, subject_id_, dataset_idx_ = next(d_iter)
            except StopIteration: # if the iterator is exhausted
                print("End of dataloader after " + str(batch * batch_size) + " samples")
                break
            
            batch_size, L, nchan, T, F = x_.shape
            sequence_center = L // 2 + 1 # index of the center element of the sequence
            
            embeddings_, _, proto_idx_, _, alphas = model.nn.get_prototypes(x_.cuda())
            
            proto_idx_ = proto_idx_.cpu().numpy()
            for c_idx, c in enumerate(channels):
                transition_matrices[c] = count_transitions(proto_idx_[:,:,c_idx], transition_matrices[c])
            
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
            groups.append(get_groups(test_datamodule, dataset_idx_, subject_id_))


        # concatenate all batches
        for c_idx, c in enumerate(channels):
            embeddings[c] = np.concatenate(embeddings[c], axis=0)
        y = np.concatenate(y, axis=0)
        subjects_id = np.concatenate(subjects_id, axis=0)
        proto_idx = np.concatenate(proto_idx, axis=0)
        groups = np.concatenate(groups, axis=0)
                    
        if len(np.unique(y)) == 5:
            label_map = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
            y = np.vectorize(label_map.get)(y)
        elif len(np.unique(y)) == 3:
            label_map = {0: "W", 1: "N", 2: "R"}
            y = np.vectorize(label_map.get)(y)
        
        # plot label ratios for each group
        plot_labels(y, groups, results_path)
        
        # plot stage ratio in each prototype and prototype counts
        prototype_density(proto_idx, n_proto, y, channels, results_path)
        
        # plot subject density in each prototype
        subject_density(proto_idx, channels, n_proto, subjects_id, results_path)
        
        # plot group density
        subject_density(proto_idx, channels, n_proto, groups, results_path, 'group')

        # plot average psd per prototype
        average_psd(channels, mean_psd, results_path)




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