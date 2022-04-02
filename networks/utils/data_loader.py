import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import xarray

tracks_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tracks/available.json')

with open(tracks_path, 'r') as ptj:
    tracks_dict = json.load(ptj)

# Taken from https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return_list = []
        for dataset in self.datasets:
            return_list.append(dataset[i])
        return return_list

    def __len__(self):
        return min(len(d) for d in self.datasets)

class CycloneDataset(Dataset):
    """
    Custom dataset for cyclones.

    TODO: Need to test
    """

    def __init__(self, cyclone_dir, transform=None, target_transform=None, target_parameters=[0,1], time_step_back=1):
        self.tracks_dict = tracks_dict
        self.cyclone_dir = cyclone_dir
        self.target_transform = target_transform
        self.time_step_back = time_step_back
        self.target_parameters = target_parameters

        
    # I think this function might be causing issues.
    def __len__(self):
        length = 0
        for cyclone, data in tracks_dict.items():
            length += len(data['coordinates'][:-2])
        
        return length            
    
    def __getitem__(self, idx):
        i = 0

        for cyclone, data in tracks_dict.items():
            j = self.time_step_back + 1
            for coordinate in data['coordinates'][:-2]:
                if i == idx:
                    
                    cyclone_ds = xarray.open_dataset(self.cyclone_dir+cyclone)
                    cyclone_ds_crop = cyclone_ds.to_array().to_numpy()
                    cyclone_ds_crop = np.transpose(cyclone_ds_crop, (1, 0, 2, 3, 4))

                    # print(f"Cyclone {cyclone} with {np.shape(cyclone_ds_crop)} and time crop {j-self.time_step_back-1} and {j}")

                    cyclone_array = cyclone_ds_crop[j-self.time_step_back-1:j,self.target_parameters,:,:,:]
                    # print(f"F1A: {cyclone_array[:,:,:,0,0]}")
                    example = torch.from_numpy(cyclone_array)
                    num_channels = int(5*len(self.target_parameters)*(1+self.time_step_back))

                    if num_channels != example.shape[0] * example.shape[1] * example.shape[2]:
                        print("Wrong shape detected")
                        
                    
                    example = torch.reshape(example, (num_channels,160,160))
                    # print(f"F1B: {example[:,0,0]}")
                    label = torch.from_numpy(np.array([[
                                                float(data['coordinates'][j-1][0]), float(data['coordinates'][j][0])], 
                                                [float(data['coordinates'][j-1][1]), float(data['coordinates'][j][1])],
                                                [float(data['categories'][j-1]), float(data['categories'][j])]
                                                        ]))
                    
                    if torch.isnan(example).any():
                        print(f"Example size is: {example.size()}")
                        print(f"Cyclone: {cyclone}")
                        print(f"Coordinate: {coordinate}")

                    return example, label

                i += 1
                j += 1

class MetaDataset(Dataset):
    """
    Custom dataset for meta data that we use to train the shallow network.

    Question at the moment is how to represent basins in the metadata?
        - 1-hot would require knowledge of all the basins... this requires writing some code to parse through the JSON.
        - As a result, we've left it out for now but it should be included later.
    """


    def __init__(self, time_step_back=1):
        self.time_step_back = time_step_back
    
    def __len__(self):
        length = 0
        
        for cyclone, data in tracks_dict.items():
            length += len(data['coordinates'][:-2])
        
        return length
    
    def __getitem__(self, idx):
        i = 0

        for cyclone, data in tracks_dict.items():
            j = self.time_step_back + 1

            for coordinate in data['coordinates'][:-2]:
                if i == idx:
                    example = torch.from_numpy(np.array([
                        float(data['categories'][j-2]),
                        float(data['categories'][j-1]),
                        float(data['coordinates'][j-2][0]),
                        float(data['coordinates'][j-2][1]),
                        float(data['coordinates'][j-1][0]),
                        float(data['coordinates'][j-1][1])
                    ]))

                    label = torch.from_numpy(np.array([[
                                                float(data['coordinates'][j-1][0]), float(data['coordinates'][j][0])], 
                                                [float(data['coordinates'][j-1][1]), float(data['coordinates'][j][1])],
                                                [float(data['categories'][j-1]), float(data['categories'][j])]
                                                        ]))

                    return example, label

                i += 1
                j += 1


def load_datasets(splits: dict):
    full_dataset_uv = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/cyclone_binaries/')
    full_length = len(full_dataset_uv)
    train_length = int(splits['train']*full_length)
    val_length = int(splits['validate']*full_length)
    test_length = full_length - train_length - val_length
    train_dataset_uv, validate_dataset_uv, test_dataset_uv = torch.utils.data.random_split(full_dataset_uv, 
        [train_length, val_length, test_length])

    full_dataset_z = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/cyclone_binaries/', target_parameters=[2])

    train_dataset_z, validate_dataset_z, test_dataset_z = torch.utils.data.random_split(full_dataset_z, 
        [train_length, val_length, test_length])

    full_dataset_meta = MetaDataset()
    meta_length = len(full_dataset_meta)

    if full_length != meta_length:
        raise Exception('Something is wrong with the meta length!')
    
    train_dataset_meta, validate_dataset_meta, test_dataset_meta = torch.utils.data.random_split(full_dataset_meta, 
        [train_length, val_length, test_length])

    full_dataset_uvz = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/cyclone_binaries/', target_parameters=[0,1,2])
    full_concat_ds = ConcatDataset(full_dataset_uvz, full_dataset_meta)

    train_concat_ds, validate_concat_ds, test_concat_ds= torch.utils.data.random_split(full_concat_ds, 
    [train_length, val_length, test_length])

    return train_dataset_uv, validate_dataset_uv, test_dataset_uv, train_dataset_z, validate_dataset_z, test_dataset_z, train_dataset_meta, validate_dataset_meta, \
            test_dataset_meta, train_concat_ds, validate_concat_ds, test_concat_ds

def get_first_example():
    # Display image and label.

    training_data = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/cyclone_binaries/', time_step_back=1)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=False)
    
    train_features, train_labels = next(iter(train_dataloader))
    # print(train_features)
    # print(train_labels)

    # print(f"Feature batch type: {type(train_features)}")
    # print(f"Labels batch type: {type(train_labels)}")
    # print(f"Feature batch size: {train_features.shape}")
    # print(f"Labels batch size: {train_labels.shape}")
    img = train_features[0][0][0][1].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.savefig('example.png')
    print(f"Label: {label}")

if __name__ == '__main__':
    splits = {'train':0.7, 'validate':0.1, 'test':0.2}
    train, val, test = load_datasets(splits)
    get_first_example()