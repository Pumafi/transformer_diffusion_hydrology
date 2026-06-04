import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

def load_training_data(path_synthetic_training_data):
    training_ndarray = np.genfromtxt(path_synthetic_training_data, delimiter=',', usemask=True, filling_values=0)
    training_data, training_mask = training_ndarray.data, training_ndarray.mask
    
    print("Loaded SAGD training data with shape: ", training_data.shape)

    return training_data, np.invert(training_mask).astype(int)


class Synthetic_Dataset(Dataset):
    def __init__(self, path_synthetic_training_data, mode="train", long_data=False, absolute_timepoints=True):
        self.history_length = 48
        self.pred_length = 24

        datafolder = './data/'
        self.test_length= 72*7
        self.valid_length = 72*5
            
        self.seq_length = self.history_length + self.pred_length

        self.absolute_timepoints = absolute_timepoints
            
        paths=datafolder+'/data.pkl' 
        sagd_observations, sagd_observations_mask = load_training_data(path_synthetic_training_data)

        self.main_data = sagd_observations
        self.mask_data = sagd_observations_mask

        self.timepoints = np.arange(self.main_data.shape[0])



        total_length = len(self.main_data)
        if mode == 'train': 
            start = 0
            end = total_length - self.seq_length - self.valid_length - self.test_length + 1
            self.use_index = np.arange(start,end,step=self.pred_length)
        if mode == 'valid': #valid
            start = total_length - self.seq_length - self.valid_length - self.test_length + self.pred_length
            end = total_length - self.seq_length - self.test_length + self.pred_length
            self.use_index = np.arange(start,end,step=self.pred_length)
        if mode == 'test': #test
            start = total_length - self.seq_length - self.test_length + self.pred_length
            end = total_length - self.seq_length + self.pred_length
            self.use_index = np.arange(start,end,step=self.pred_length)

            if long_data:
                self.use_index = np.array([start, ])
                self.seq_length = total_length - start 



    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index:index+self.seq_length].copy()
        target_mask[-self.pred_length:] = 0. #pred mask for test pattern strategy
        
        if self.absolute_timepoints :
            timepoints = self.timepoints[index:index+self.seq_length]
        else:
            timepoints = np.arange(self.seq_length) * 1.0

        s = {
            'observed_data': self.main_data[index:index+self.seq_length],
            'observed_mask': self.mask_data[index:index+self.seq_length],
            'gt_mask': target_mask,
            'timepoints': timepoints,
            'feature_id': np.arange(self.main_data.shape[1]) * 1.0, 
        }

        return s
    def __len__(self):
        return len(self.use_index)

def get_dataloader(path_synthetic_training_data, device, batch_size=8, long_data=False, absolute_timepoints=True):
    dataset = Synthetic_Dataset(path_synthetic_training_data, mode='train', absolute_timepoints=absolute_timepoints)
    print("Training dataset len: ", len(dataset))
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Synthetic_Dataset(path_synthetic_training_data, mode='valid', absolute_timepoints=absolute_timepoints)
    print("Valid dataset len: ", len(valid_dataset))
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Synthetic_Dataset(path_synthetic_training_data, mode='test', long_data=long_data, absolute_timepoints=absolute_timepoints)
    print("Test dataset len: ", len(test_dataset))
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)

    #scaler = torch.from_numpy(dataset.std_data).to(device).float()
    #mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return train_loader, valid_loader, test_loader#, scaler, mean_scaler