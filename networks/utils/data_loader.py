import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import json
import matplotlib.pyplot as plt

with open('../../tracks/avaliable.json', 'r') as ptj:
    tracks_dict = json.load(ptj)

class CycloneDataset(Dataset):
    """
    Custom dataset for cyclones.
    """

    def __init__(self, cyclone_dir, transform=None, target_transform=None, time_step_back=1):
        self.tracks_dict = tracks_dict
        self.cyclone_dir = cyclone_dir
        self.transform = transform
        self.target_transform = target_transform
        self.time_step_back = time_step_back

    def __len__(self):
        length = 0
        for cyclone, data in tracks_dict.items():
            length += len(data['coordinates'])
        
        return length

    def __getitem__(self, idx):
        i = 0

        for cyclone, data in tracks_dict.items():
            j = self.time_step_back
            for coordinate in data['coordinates']:
                if i == idx:
                    cyclone_array = np.load(self.cyclone_dir+cyclone)
                    example = torch.from_numpy(cyclone_array[j-self.time_step_back:j+1,:,:,:,:])
                    label = torch.from_numpy(np.array([float(data['coordinates'][j][0]), float(data['coordinates'][j][1]), float(data['categories'][j])]))
                    return example, label

                i += 1
                j += 1
        
def load_datasets(splits: dict):
    full_dataset = CycloneDataset(cyclone_dir='/g/data/x77/jm0124/cyclone_binaries/')
    full_length = len(full_dataset)
    train_length = int(splits['train']*full_length)
    val_length = int(splits['validate']*full_length)
    test_length = full_length - train_length - val_length
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(full_dataset, 
        [train_length, val_length, test_length])

    return train_dataset, validate_dataset, test_dataset

def get_first_example():
    # Display image and label.

    training_data = CycloneDataset(cyclone_dir='/g/data/x77/jm0124/cyclone_binaries/', time_step_back=1)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=False)
    
    train_features, train_labels = next(iter(train_dataloader))
    print(train_features)
    print(train_labels)

    print(f"Feature batch type: {type(train_features)}")
    print(f"Labels batch type: {type(train_labels)}")
    print(f"Feature batch size: {train_features.shape}")
    print(f"Labels batch size: {train_labels.shape}")
    img = train_features[0][0][0][0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.savefig('example.png')
    print(f"Label: {label}")

if __name__ == '__main__':
    splits = {'train':0.7, 'validate':0.1, 'test':0.2}
    train, val, test = load_datasets(splits)
    get_first_example()