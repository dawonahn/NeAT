import os
import torch
import numpy as np
from pathlib import Path
from dotmap import DotMap
from torch.utils.data import Dataset



class COODataset(Dataset):
    def __init__(self, idxs, vals):
        self.idxs = idxs
        self.vals = vals

    def __len__(self):
        return self.vals.shape[0]

    def __getitem__(self, idx):
        return self.idxs[idx], self.vals[idx]


def read_data(cfg, bin_val=True, neg=True):
    ''' Read tensors in COO format.
        cfg: configuration file
        bin_val: whether if tensor values are binary
        neg: include negatvie sampling
    '''

    dct = DotMap()
    name = cfg.dataset
    device = cfg.device
    data_path = os.path.join(Path.home(), 'TENSOR', name)

    for dtype in ['train', 'valid', 'test']:
        if cfg.verbose:
            print(f"Reading {dtype} dataset----------------------")

        idxs_lst = []
        vals_lst = []

        idxs = np.load(f'{data_path}/{dtype}_idxs.npy')
        if bin_val:
            vals = torch.ones(idxs.shape[0])
        else:
            vals = np.load(f'{data_path}/{dtype}_vals.npy')

        idxs_lst.append(idxs)
        vals_lst.append(vals)
        
        if neg:
            if cfg.verbose:
                print(f"Read negative samples")
            neg_path = os.path.join(data_path, 'neg_sample0')
            neg_idxs = np.load(f'{neg_path}/{dtype}_idxs.npy')
            neg_vals = np.zeros(neg_idxs.shape[0])
            idxs_lst.append(neg_idxs)
            vals_lst.append(neg_vals)
    
        total_idxs = np.vstack(idxs_lst)
        total_vals = np.hstack(vals_lst)

        dct[f'{dtype}_i'] = torch.LongTensor(total_idxs).to(device)
        dct[f'{dtype}_v'] = torch.FloatTensor(total_vals).to(device)   
            
    dct.sizes = get_size(name)
    print(f"DATASET: {cfg.dataset} "
          f"|| size: {dct.sizes} & training nnz: {dct[f'train_v'].shape[0]}")    
    
    return dct
    

def get_size(name):
    '''
    Get size (dimensionality) of tensor.
    name: dataset name
    '''
    
    if name == "ml": 
        size = [610, 9724, 4110]
        
    if name == "yelp": 
        size = [70818, 15580, 109]
        
    if name == "foursquare_nyc":
        size = [1084, 38334, 7641]
        
    if name == "foursquare_tky":
        size = [2294, 61859, 7641]
        
    if name == "yahoo_msg":
        size = [82309, 82308, 168]

    if name.endswith('dblp'):
        size = [4057, 14328, 7723]


    return size
