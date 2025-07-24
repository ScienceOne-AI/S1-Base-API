PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25

import numpy as np
import torch
import os
from typing import Optional, List, Tuple

class SurfacePressureDataset(torch.utils.data.Dataset):

    def __init__(self, data_list: list, num_points: int, repeat: int):
        """
        Initializes the SurfacePressureDataset instance.

        Args:
            data_list: The list of data used for finetuning.
            num_points: Fixed number of points to sample from each 3D model.
            repeat: times for repeat the samples in dataset.
        """
        if len(data_list) == 0:
            self.data_list = [os.path.join('data/', name) for name in os.listdir('data/')]
        else:
            self.data_list = data_list
            
        self.pc = [np.loadtxt(data_path) for data_path in self.data_list]
        self.num_points = num_points
        self.repeat = repeat

    def __len__(self):
        return len(self.data_list) * self.repeat

    def __getitem__(self, idx):
        point_cloud = self.pc[idx//self.repeat].copy()
        point_cloud_tensor = torch.tensor(point_cloud[:,:3], dtype=torch.float32)
        point_cloud_tensor[:, 0] = point_cloud_tensor[:, 0] - (torch.max(point_cloud_tensor[:, 0]) + torch.min(point_cloud_tensor[:, 0]))/2
        
        pressures_tensor = torch.tensor(point_cloud[:,3], dtype=torch.float32)
        
        N = point_cloud.shape[0]
        rand_idx = torch.randperm(N)
        point_cloud_tensor = point_cloud_tensor[rand_idx, :][:self.num_points]  # [B, N, 3]
        pressures_tensor = pressures_tensor[rand_idx][:self.num_points]

        env = self.data_list[idx//self.repeat].split('_')[-2:]
        v = float(env[0])
        w = float(env[1][:-4])
        scale = torch.max(torch.abs(point_cloud_tensor))
        point_cloud_tensor = point_cloud_tensor / scale
        return point_cloud_tensor, pressures_tensor, torch.tensor([v, w, scale]) 



class SurfacePressureDatasetWeb(torch.utils.data.Dataset):

    def __init__(self, data_arrays: list[np.ndarray], speed_params: List[Tuple[float, float, float]], num_points: int, repeat: int):
        """
        Initializes the SurfacePressureDataset instance.

        Args:
            data_arrays: list of np.ndarray, each shape (N,4): x,y,z,p_true
            speed_params: list of (v, w_y, w_z) per sample
            num_points: Fixed number of points to sample from each 3D model.
            repeat: times for repeat the samples in dataset.
        """

        assert all(arr.shape[1] == 4 for arr in data_arrays), "每个数组必须有 4 列 (x,y,z,p_true)"
            
        self.pc = data_arrays
        self.num_points = num_points
        self.repeat = repeat
        self.speed_params = speed_params

    def __len__(self):
        return len(self.pc) * self.repeat

    def __getitem__(self, idx):
        point_cloud = self.pc[idx//self.repeat].copy()
        v, w_y, w_z = self.speed_params[idx//self.repeat]
        point_cloud_tensor = torch.tensor(point_cloud[:,:3], dtype=torch.float32)
        point_cloud_tensor[:, 0] = point_cloud_tensor[:, 0] - (torch.max(point_cloud_tensor[:, 0]) + torch.min(point_cloud_tensor[:, 0]))/2
        
        pressures_tensor = torch.tensor(point_cloud[:,3], dtype=torch.float32)
        
        N = point_cloud.shape[0]
        rand_idx = torch.randperm(N)
        point_cloud_tensor = point_cloud_tensor[rand_idx, :][:self.num_points]  # [B, N, 3]
        pressures_tensor = pressures_tensor[rand_idx][:self.num_points]

        scale = torch.max(torch.abs(point_cloud_tensor))
        point_cloud_tensor = point_cloud_tensor / scale
        return point_cloud_tensor, pressures_tensor, torch.tensor([v, w_y, scale]) 