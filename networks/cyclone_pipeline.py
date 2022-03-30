from utils.loss_functions import L2_Dist_Func_Intensity
from utils.data_loader import *
import torch
from datetime import datetime
from models.uv_model import UV_Model
from models.z_model import Z_Model
from models.meta_model import Meta_Model
import os
import datetime
import xarray

data_dir = '/g/data/x77/ob2720/cyclone_binaries/'
models_dir = '/g/data/x77/jm0124/models'

def train_single_models_epoch(model, epoch, train_dataloader, loss_func, optimizer):
    
    running_loss = 0
    last_loss = 0

    for i, (example, truth) in enumerate(train_dataloader):
        optimizer.zero_grad()

        output = model(example)

        loss = loss_func(output, truth)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0
        
    print("Reached end of train single models")
    
    return last_loss

def train_component(model, train_dataloader, val_dataloader, loss_fn, optimizer, model_name, num_epochs):
    print(f"-------- Training {model_name} model --------")

    best_vloss = 1e10
    
    for epoch in range(0,num_epochs):
        print(f"EPOCH: {epoch}")

        model.train(True)
        avg_loss = train_single_models_epoch(model, epoch, 
            train_dataloader, loss_fn, optimizer)
        
        model.train(False)

        running_vloss = 0

        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
        
        avg_vloss = running_vloss / (i+1)

        print(f"Loss train for {model_name} {avg_loss} valid {avg_vloss}")

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'{models_dir}/{model_name}'
            torch.save(model.state_dict(), model_path)

    print(f"Best vloss for {model_name} was {best_vloss}")

def train_single_models(train_dataloader_uv, val_dataloader_uv, train_dataloader_z, val_dataloader_z, train_dataloader_meta,
    val_dataloader_meta, learning_rate, betas, eps, weight_decay):
    
    loss_fn = L2_Dist_Func_Intensity()
    
    print("-------- Loading models --------")

    model_uv = UV_Model()

    # if 'model_uv' in os.listdir(models_dir):
    #     model_uv.load_state_dict(torch.load(f'{models_dir}/model_uv'))
    
    print("Model UV loaded")
    
    model_z = Z_Model()

    # if 'model_z' in os.listdir(models_dir):
    #     model_z.load_state_dict(torch.load(f'{models_dir}/model_z'))
    
    print("Model z loaded")

    model_meta = Meta_Model()

    # if 'model_z' in os.listdir(models_dir):
    #     model_meta.load_state_dict(torch.load(f'{models_dir}/model_meta'))
    
    print("Model meta loaded")

    # Here we are using the same optimizer, but we could simply change the learning rate
    # in order to accomodate the differences needed for different networks.

    optimizer = torch.optim.Adam(model_uv.parameters(), lr=learning_rate, betas=betas, eps=1e-8,
                           weight_decay=weight_decay)

    EPOCHS = 2

    train_component(model_uv, train_dataloader_uv, val_dataloader_uv, loss_fn, optimizer, "model_uv", EPOCHS)
    # train_component(model_z, train_dataloader_z, val_dataloader_z, loss_fn, optimizer, "model_z", EPOCHS)
    # train_component(model_meta, train_dataloader_meta, val_dataloader_meta, loss_fn, optimizer, "model_meta", EPOCHS)

if __name__ == '__main__':
    splits = {'train':0.7, 'validate':0.1, 'test':0.2}
    train_dataset_uv, validate_dataset_uv, test_dataset_uv, train_dataset_z, validate_dataset_z, test_dataset_z, train_dataset_meta, validate_dataset_meta, test_dataset_meta = load_datasets(splits)

    training_loader_uv = torch.utils.data.DataLoader(train_dataset_uv, batch_size=10, shuffle=False, num_workers=1, drop_last=True)
    validation_loader_uv = torch.utils.data.DataLoader(validate_dataset_uv, batch_size=10, shuffle=False, num_workers=1, drop_last=True)

    training_loader_z = torch.utils.data.DataLoader(train_dataset_z, batch_size=5, shuffle=False, num_workers=1, drop_last=True)
    validation_loader_z = torch.utils.data.DataLoader(validate_dataset_z, batch_size=5, shuffle=False, num_workers=1, drop_last=True)

    meta_training_loader = torch.utils.data.DataLoader(train_dataset_meta, batch_size=10, shuffle=False, num_workers=1, drop_last=True)
    meta_validation_loader = torch.utils.data.DataLoader(validate_dataset_meta, batch_size=10, shuffle=False, num_workers=1, drop_last=True)

    train_single_models(training_loader_uv, validation_loader_uv, training_loader_z, validation_loader_z, meta_training_loader,
        meta_validation_loader, 1e-3, (0.9, 0.999), 1e-8, 1e-4)