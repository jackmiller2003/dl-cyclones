import torch
from torch.utils.data import Dataset
import os
import numpy as np

with open('../tracks/proc_tracks.json', 'r') as ptj:
    tracks_dict = json.load(ptj)

class CycloneDataset(Dataset):
    def __init__(self, cyclone_dir, transform=None, target_transform=None):
        self.tracks_dict = tracks_dict
        self.cyclone_dir = cyclone_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        length = 0
        for cyclone in tracks_dict:
            length += len(cyclone['iso_times'])
        
        return length

    def __getitem__(self, idx):
        i = 0

        for cyclone in tracks_dict:
            j = 0
            for coordinate in cyclone['coordinates']:
                if i == idx:
                    cyclone_array = np.load(cyclone_dir+cyclone)
                    example = cyclone_array[j,:,:,:,:]
                    label = (cyclone['coordinates'][j], cyclone['categories'][j])
                    break

                i += 1
                j += 1

if __name__ == '__main__':
    # Display image and label.

    training_data = CycloneDataset('/g/data/x77/jm0124/cyclone_binaries/')
    train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)

    for i in range(0,50000):
        try:
            train_features, train_label = train_dataloader[i]
        except:
            continue

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")