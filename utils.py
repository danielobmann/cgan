import os
import torch
import torchvision.transforms as T
from glob import glob
from PIL import Image
import numpy as np


# data can be downloaded from https://drive.grand-challenge.org/
class DriveDataset(torch.utils.data.Dataset):
    def __init__(self, folder, size=128):
        if "manual" in folder:
            self.filelist = glob(os.path.join(folder, "*.gif"))
        else:
            self.filelist = glob(os.path.join(folder, "*manual/*.gif"))
        self.transform = T.Compose([T.RandomPerspective(p=1.0), T.RandomRotation(30),
                                    T.RandomResizedCrop(size, scale=(0.1, 0.5)),
                                    T.ColorJitter(), T.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.filelist[index])
        return self.transform(img)

    def __len__(self):
        return len(self.filelist)


def get_data(folder="data/training/", batch_size=16, val_split=0.3, shuffle=True, seed=None):
    dataset = DriveDataset(folder=folder)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, val_loader