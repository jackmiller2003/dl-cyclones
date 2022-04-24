import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Pool
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import xarray

tracks_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tracks/available.json')
test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tracks/test.json')
one_hot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tracks/one_hot_dict.json')

with open(tracks_path, 'r') as ptj:
    tracks_dict = json.load(ptj)

with open(one_hot_path, 'r') as oht:
    one_hot_dict = json.load(oht)

# with open(test_path, 'r') as test_json:
#     test_dict = json.load(test_json)


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

    def __init__(self, cyclone_dir, transform=None, target_transform=None, target_parameters=[0,1], time_step_back=1, day_pred=True, tracks_used = tracks_dict):
        self.tracks_dict = tracks_used
        self.cyclone_dir = cyclone_dir
        self.target_transform = target_transform
        self.time_step_back = time_step_back
        self.target_parameters = target_parameters
        self.day_predict = day_pred

        
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
                        
                        cyclone_ds = xarray.open_dataset(self.cyclone_dir+cyclone)                    
                        cyclone_ds_new = cyclone_ds[dict(time=list(range(j-self.time_step_back-1,j)))]
                        
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


def load_datasets(splits: dict):
    
    full_dataset_uv = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/cyclone_binaries/')
    full_length = len(full_dataset_uv)
    train_length = int(splits['train']*full_length)
    val_length = int(splits['validate']*full_length)
    test_length = full_length - train_length - val_length
    
    torch.manual_seed(0)
    train_dataset_uv, validate_dataset_uv, test_dataset_uv = torch.utils.data.random_split(full_dataset_uv, 
        [train_length, val_length, test_length], generator=torch.Generator().manual_seed(0))
    torch.manual_seed(torch.initial_seed())

    full_dataset_z = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/cyclone_binaries/', target_parameters=[2])

    torch.manual_seed(0)
    train_dataset_z, validate_dataset_z, test_dataset_z = torch.utils.data.random_split(full_dataset_z, 
        [train_length, val_length, test_length], generator=torch.Generator().manual_seed(0))
    torch.manual_seed(torch.initial_seed())

    full_dataset_meta = MetaDataset()
    meta_length = len(full_dataset_meta)

    # if full_length != meta_length:
    #     raise Exception('Something is wrong with the meta length!')
    
    torch.manual_seed(0)
    train_dataset_meta, validate_dataset_meta, test_dataset_meta = torch.utils.data.random_split(full_dataset_meta, 
        [train_length, val_length, test_length], generator=torch.Generator().manual_seed(0))
    torch.manual_seed(torch.initial_seed())

    full_dataset_uvz = CycloneDataset(cyclone_dir='/g/data/x77/ob2720/cyclone_binaries/', target_parameters=[0,1,2])
    full_concat_ds = ConcatDataset(full_dataset_uvz, full_dataset_meta)

    torch.manual_seed(0)
    train_concat_ds, validate_concat_ds, test_concat_ds= torch.utils.data.random_split(full_concat_ds, 
    [train_length, val_length, test_length], generator=torch.Generator().manual_seed(0))
    torch.manual_seed(torch.initial_seed())

    return train_dataset_uv, validate_dataset_uv, test_dataset_uv, train_dataset_z, validate_dataset_z, test_dataset_z, train_dataset_meta, validate_dataset_meta, \
            test_dataset_meta, train_concat_ds, validate_concat_ds, test_concat_ds


def load_holdout_test():
    holdout_test_dataset_uv = CycloneDataset(cyclone_dir='/g/data/x77/jm0124/test_holdout/', tracks_used=test_dict)
    holdout_test_dataset_z = CycloneDataset(cyclone_dir='/g/data/x77/jm0124/test_holdout/', target_parameters=[2], tracks_used=test_dict)
    holdout_test_dataset_meta = CycloneDataset(tracks_used=test_dict)
    

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