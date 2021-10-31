from __future__ import print_function, division
import numpy as np
import os
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset



def get_data_loaders(data_dir,
                     batch_size,
                     train_transform,
                     test_transform,
                     shuffle=True,
                     num_workers=4,
                     pin_memory=False):
    """
    Adapted from: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    
    Utility function for loading and returning train and test
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, set pin_memory to True.
    
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - train_transform: pytorch transforms for the training set
    - test_transform: pytorch transofrms for the test set
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    
    Returns
    -------
    - train_loader: training set iterator.
    - test_loader:  test set iterator.
    """
    
    # Load the datasets
    train_dataset = CheXpert_Dataset(csv_file='../../../data/CheXpert-v1.0/train.csv',root_dir='../../../data',transform=train_transform)
    test_dataset= CheXpert_Dataset(csv_file='../../../data/CheXpert-v1.0/valid.csv',root_dir='../../../data',transform=test_transform)
 
    # Create loader objects
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )
          
    return (train_loader, test_loader)
class CheXpert_Dataset(Dataset):


    def __init__(self, csv_file, root_dir, transform=None):
        """
#        Args:
#            csv_file (string): Path to the csv file with annotations.
#            root_dir (string): Directory with all the images.
#            transform (callable, optional): Optional transform to be applied
#                on a sample.
        """
        self.diseases = pd.read_csv(csv_file).fillna(0.5)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.diseases)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.diseases.iloc[idx, 0])
        image = Image.open(img_name)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        diseases = self.diseases.iloc[idx:idx+1,8:-1]
        diseases = np.array(diseases)
        diseases =torch.from_numpy(diseases)[0]
        return image,diseases
