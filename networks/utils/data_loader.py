import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Pool
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import xarray
from pathlib import Path

tracks_path = str(Path(__file__).parent.parent.parent / 'tracks' / 'available.json')
one_hot_path = str(Path(__file__).parent.parent.parent / 'tracks' / 'one_hot_dict.json')

train_json_path = '/g/data/x77/ob2720/partition/train.json'
valid_json_path = '/g/data/x77/ob2720/partition/valid.json'
test_json_path = '/g/data/x77/ob2720/partition/test.json'

with open(train_json_path, 'r') as tj:
    train_dict = json.load(tj)

with open(valid_json_path, 'r') as vj:
    val_dict = json.load(vj)

with open(test_json_path, 'r') as tj:
    test_dict = json.load(tj)

with open(tracks_path, 'r') as ptj:
    tracks_dict = json.load(ptj)

with open(one_hot_path, 'r') as oht:
    one_hot_dict = json.load(oht)

# Taken from https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return_list = []
        examples = []
        for dataset in self.datasets:
            example, label = dataset[i]
            examples.append(example)
        return examples, label

    def __len__(self):
        return min(len(d) for d in self.datasets)

class CycloneDataset(Dataset):
    """
    Custom dataset for cyclones.
    """

    def __init__(self, cyclone_dir, transform=None, target_transform=None, target_parameters=[0,1], time_step_back=1, day_pred=True, tracks_used = tracks_dict):
        self.tracks_dict = tracks_used
        self.cyclone_dir = cyclone_dir
        self.target_transform = target_transform
        self.time_step_back = time_step_back
        self.target_parameters = target_parameters
        self.day_predict = True

        
    # I think this function might be causing issues.
    def __len__(self):
        length = 0
        if self.day_predict:
            for cyclone, data in self.tracks_dict.items():
                if len(data['coordinates']) > 9:
                    length += len(data['coordinates'][:-9])
            return length
        else:
            for cyclone, data in self.tracks_dict.items():
                if len(data['coordinates']) > 2:
                    length += len(data['coordinates'][:-2])
            return length            
    
    def __getitem__(self, idx):
        i = 0

        for cyclone, data in self.tracks_dict.items():
            j = self.time_step_back + 1

            if self.day_predict:
                bound = 9
            else:
                bound = 2

            if len(data['coordinates']) > bound:
                for coordinate in data['coordinates'][:-bound]:
                    if i == idx:
                        cyclone_ds = xarray.open_dataset(self.cyclone_dir+cyclone+".nc", engine='netcdf4')

                        try:
                            cyclone_ds_new = cyclone_ds[dict(time=list(range(j-self.time_step_back-1,j)))]
                        except Exception as e:
                            raise print(f"{cyclone} {e}")
                        
                        if self.target_parameters == [0,1]:
                            cyclone_ds_new = cyclone_ds_new[['u','v']]
                        elif self.target_parameters == [2]:
                            cyclone_ds_new = cyclone_ds_new[['z']]
                        elif self.target_parameters == [0,1,2]:
                            cyclone_ds_new = cyclone_ds_new[['u','v', 'z']]
                        
                        cyclone_ds_crop_new = cyclone_ds_new.to_array().to_numpy()
                        cyclone_ds_crop_new = np.transpose(cyclone_ds_crop_new, (1, 0, 2, 3, 4))
                        
                        example = torch.from_numpy(cyclone_ds_crop_new)
                        num_channels = int(5*len(self.target_parameters)*(1+self.time_step_back))             
                        example = torch.reshape(example, (num_channels,160,160))

                        if self.day_predict:

                            label = torch.from_numpy(np.array([[
                                                        float(data['coordinates'][j-1][0]), float(data['coordinates'][j+bound-2][0])], 
                                                        [float(data['coordinates'][j-1][1]), float(data['coordinates'][j+bound-2][1])],
                                                        [float(data['categories'][j-1]), float(data['categories'][j])]
                                                                ]))
                        else:
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


    def __init__(self, time_step_back=1, day_pred=True, tracks_used=tracks_dict):
        self.time_step_back = time_step_back
        self.day_predict = True
        self.tracks_dict = tracks_used
    
    def __len__(self):
        length = 0
        if self.day_predict:
            for cyclone, data in self.tracks_dict.items():
                if len(data['coordinates']) > 9:
                    length += len(data['coordinates'][:-9])
            return length
        else:
            for cyclone, data in self.tracks_dict.items():
                if len(data['coordinates']) > 2:
                    length += len(data['coordinates'][:-2])
            return length  
    
    def __getitem__(self, idx):
        i = 0

        for cyclone, data in self.tracks_dict.items():
            j = self.time_step_back + 1

            if self.day_predict:
                bound = 9
            else:
                bound = 2

            if len(data['coordinates']) > bound:
                for coordinate in data['coordinates'][:-bound]:
                    if i == idx:
                        
                        sub_basin_encoding = np.zeros((9,1))
                        sub_basin_encoding[one_hot_dict[data['subbasin'][j-1]]] = 1

                        example = torch.from_numpy(np.array([
                            float(data['categories'][j-2]),
                            float(data['categories'][j-1]),
                            float(data['coordinates'][j-2][0]),
                            float(data['coordinates'][j-2][1]),
                            float(data['coordinates'][j-1][0]),
                            float(data['coordinates'][j-1][1])
                        ]))

                        # Size is now 6 + 9 = 15
                        example = np.append(example, sub_basin_encoding)

                        if self.day_predict:

                            label = torch.from_numpy(np.array([[
                                                        float(data['coordinates'][j-1][0]), float(data['coordinates'][j+bound-2][0])], 
                                                        [float(data['coordinates'][j-1][1]), float(data['coordinates'][j+bound-2][1])],
                                                        [float(data['categories'][j-1]), float(data['categories'][j])]
                                                                ]))
                        else:
                            label = torch.from_numpy(np.array([[
                                                        float(data['coordinates'][j-1][0]), float(data['coordinates'][j][0])], 
                                                        [float(data['coordinates'][j-1][1]), float(data['coordinates'][j][1])],
                                                        [float(data['categories'][j-1]), float(data['categories'][j])]
                                                                ]))

                        return example, label

                    i += 1
                    j += 1

def load_json(fname):
    import json
    with open(fname, 'r') as f:
        return json.load(f)

def load_datasets():
    train_dataset_uv = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/partition/train/', tracks_used=train_dict, target_parameters=[0,1])
    valid_dataset_uv = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/partition/valid/', tracks_used=val_dict, target_parameters=[0,1])
    test_dataset_uv  = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/partition/test/',  tracks_used=test_dict, target_parameters=[0,1])

    train_dataset_z = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/partition/train/', tracks_used=train_dict, target_parameters=[2])
    valid_dataset_z = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/partition/valid/', tracks_used=val_dict, target_parameters=[2])
    test_dataset_z  = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/partition/test/',  tracks_used=test_dict, target_parameters=[2])

    train_dataset_meta = MetaDataset(tracks_used=load_json('/g/data/x77/ob2720/partition/train.json'))
    valid_dataset_meta = MetaDataset(tracks_used=load_json('/g/data/x77/ob2720/partition/valid.json'))
    test_dataset_meta  = MetaDataset(tracks_used=load_json('/g/data/x77/ob2720/partition/test.json'))

    train_dataset_uvz = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/partition/train/', tracks_used=train_dict, target_parameters=[0,1,2])
    valid_dataset_uvz = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/partition/valid/', tracks_used=val_dict, target_parameters=[0,1,2])
    test_dataset_uvz  = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/partition/test/',  tracks_used=test_dict, target_parameters=[0,1,2])

    train_concat_ds = ConcatDataset(train_dataset_uvz, train_dataset_meta)
    valid_concat_ds = ConcatDataset(valid_dataset_uvz, valid_dataset_meta)
    test_concat_ds =  ConcatDataset(test_dataset_uvz,  test_dataset_meta)

    return train_dataset_uv,   valid_dataset_uv,   test_dataset_uv, \
           train_dataset_z,    valid_dataset_z,    test_dataset_z, \
           train_dataset_meta, valid_dataset_meta, test_dataset_meta, \
           train_concat_ds,    valid_concat_ds,    test_concat_ds

def get_first_example():
    # Display image and label.

    training_data = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/cyclone_binaries/', time_step_back=1)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=False)
    
    train_features, train_labels = next(iter(train_dataloader))
    img = train_features[0][0][0][1].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.savefig('example.png')
    print(f"Label: {label}")

if __name__ == '__main__':
    splits = {'train':0.7, 'validate':0.1, 'test':0.2}
    train, val, test = load_datasets(splits)
    get_first_example()