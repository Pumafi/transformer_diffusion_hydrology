import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

def load_training_data(path_sagd_training_data, path_covariate_data=None):
    training_ndarray = np.genfromtxt(path_sagd_training_data, delimiter=',', usemask=True, filling_values=0)
    training_data, training_mask = training_ndarray.data, training_ndarray.mask
    if path_covariate_data is not None:
        covariates = np.genfromtxt(path_covariate_data, delimiter=',', usemask=True, filling_values=0)
    else:
        covariates = np.zeros(training_data.shape, dtype=training_data.dtype)
        
    print("Loaded SAGD training data with shape: ", training_data.shape)
    print("Loaded SAGD training mask with shape: ", training_mask.shape)
    print("Loaded covariates with shape: ", covariates.shape)

    return training_data, np.invert(training_mask).astype(int), covariates


class OPE_Dataset(Dataset):
    def __init__(self, path_sagd_training_data, path_covariate_data=None, mode="train", long_data=False, safran_covariates=False, absolute_timepoints=True):
        # Define lengths for history, prediction, validation, and testing
        # TODO: These can be made configurable if needed, but for now we will keep them fixed for simplicity.

        self.history_length = 18
        self.pred_length = 6
        self.test_length= 24*7
        self.valid_length = 24*5
        
        self.seq_length = self.history_length + self.pred_length

        self.use_safran_covariates = safran_covariates
        self.absolute_timepoints = absolute_timepoints
            
        sagd_observations, sagd_observations_mask, covariates = load_training_data(path_sagd_training_data, path_covariate_data)

        # Set and normalize covariates even if not specified, so only one class required for dataset.
        # If not using covariates, they will just be ignored in the __getitem__ method.
        self.covariates = covariates
        self.mean_covariates, self.std_covariates = self.covariates.mean(axis=0), self.covariates.std(axis=0)
        self.covariates = (self.covariates - self.mean_covariates) / self.std_covariates

        # Set main data and mask
        self.main_data = sagd_observations
        self.mask_data = sagd_observations_mask

        # Create timepoints for absolute time encoding if needed
        self.timepoints = np.arange(self.main_data.shape[0])

        self.main_data = self.main_data
        self.mask_data = self.mask_data
        
        # Normalize main data
        self.mean_data, self.std_data = self.main_data.mean(axis=0), self.main_data.std(axis=0)
        self.main_data = (self.main_data - self.mean_data) / self.std_data

        # Utilize a sliding window approach to create sequences for training, validation, and testing
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
                # This is to test long-horizon forecasting, where we return only one long test sequence instead of multiple shorter test sequences.
                self.use_index = np.array([start, ])
                self.seq_length = total_length - start 



    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index:index+self.seq_length].copy()
        target_mask[-self.pred_length:] = 0. # pred mask for test pattern strategy, basically forecasting mask that can be used
        
        # For absolute time encoding, we return the actual timepoints. For relative time encoding, we return a range from 0 to seq_length-1.
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
        # Only include covariates if specified
        if self.use_safran_covariates:
            covariates = self.covariates[index:index+self.seq_length]
            s['covariates'] = covariates

        return s
        
    def __len__(self):
        return len(self.use_index)

def get_dataloader(path_sagd_training_data, device, path_covariate_data=None, batch_size=8, long_data=False, safran_covariates=False, absolute_timepoints=True, silent=False):
    
    dataset = OPE_Dataset(path_sagd_training_data, path_covariate_data, mode='train', safran_covariates=safran_covariates, absolute_timepoints=absolute_timepoints)
    
    if not silent:
        print("Training dataset len: ", len(dataset))
        
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    
    valid_dataset = OPE_Dataset(path_sagd_training_data, path_covariate_data, mode='valid', safran_covariates=safran_covariates, absolute_timepoints=absolute_timepoints)
    
    if not silent:
        print("Valid dataset len: ", len(valid_dataset))
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    
    test_dataset = OPE_Dataset(path_sagd_training_data, path_covariate_data, mode='test', long_data=long_data, safran_covariates=safran_covariates, absolute_timepoints=absolute_timepoints)

    if not silent:
        print("Test dataset len: ", len(test_dataset))
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)

    # These are not used, but we return them for potential use in scaling the model outputs back to the original scale if needed.
    scaler = torch.from_numpy(dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler