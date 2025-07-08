from typing import Callable, List, Tuple
import os 
import shutil
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd

class GroupPredictionConverter():

    def __init__(
        self,
        dataset_name: str,
        data_folder: str,
        exclude: List[str] = [], 
    ):
        self.dataset_name = dataset_name
        self.data_folder = data_folder
        self.dataset_folder = Path(os.path.join(data_folder, dataset_name))
        self.exclude = [Path(p) for p in exclude]
        self.exclude.append(self.dataset_folder / 'labels')
        self.new_dataset_folder = Path(str(self.dataset_folder).replace(self.dataset_name, self.dataset_name + '_group_prediction'))
        os.makedirs(self.new_dataset_folder, exist_ok=True)

        self.table = pd.read_csv(self.dataset_folder / 'table.csv', index_col=0)

        assert os.path.exists(self.dataset_folder), "{dataset_name} not in {data_folder}"
        assert os.path.isfile(self.dataset_folder / 'table.csv'), "'table.csv' not in {self.dataset_folder}"
        # assert os.path.exists(self.dataset_folder / 'raw'), "/raw not in {self.dataset_folder}"


    def __call__(self):

        full_paths = list(self.dataset_folder.glob("*"))

        # copy data to new folder
        for sd in tqdm(full_paths, desc="Copying data"):
            if sd not in self.exclude:
                sd_copy = Path(str(sd.parent).replace(self.dataset_name, self.dataset_name + '_group_prediction')) / sd.name
                os.makedirs(sd_copy.parent, exist_ok=True)

                if os.path.isfile(sd):
                    shutil.copy(sd, sd_copy)
                else: 
                    shutil.copytree(sd, sd_copy)

        g_dict = {
            'group': [],
            'label': []
        }

        for g_idx, g in enumerate(self.table['group'].unique()):
            g_dict['group'].append(g)
            g_dict['label'].append(g_idx)
        g_df = pd.DataFrame(g_dict)
        g_df.to_csv(self.new_dataset_folder / 'group_labels')

        os.makedirs(self.new_dataset_folder / 'labels', exist_ok=True)
        for l_file in (self.dataset_folder / 'labels').glob('*'):
            subject_id = l_file.name[:-4]
            num_windows = self.table[self.table['subject_id'] == int(subject_id)]['num_windows'].iloc[0]
            group = self.table[self.table['subject_id'] == int(subject_id)]['group'].iloc[0]
            
            y = np.full(num_windows, g_df[g_df['group'] == group]['label'].iloc[0])

            l_path = self.new_dataset_folder / 'labels' / l_file.name

            labels_memmap = np.memmap(
                l_path, dtype=np.int16, mode="w+", shape=num_windows
            )
            labels_memmap[:] = y[:]
            labels_memmap.flush()


if __name__ == "__main__":

    c = GroupPredictionConverter('alzheimers', data_folder="/esat/biomeddata/guests/JavierGarcia")
    c()

