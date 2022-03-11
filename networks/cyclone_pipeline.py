from utils.loss_functions import L2_Dist_Func_Intensity
from utils.data_loader import *
import torch
from models.uv_model import UV_Model
from models.meta_model import Meta_Model
import os
import datetime

data_dir = '/g/data/x77/jm0124/cyclone_binaries'
models_dir = '/g/data/x77/jm0124/models'

def train_single_models_epoch(model, epoch, train_dataloader, loss_func, optimizer):
    
    running_loss = 0
    last_loss = 0

    for i, (example, truth) in enumerate(train_dataloader):
        optimizer.zero_grad()

        outputs = model(example)

        loss = loss_func(output, truth)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0
        
    return last_loss

def train_single_models(train_dataloader, val_dataloader, learning_rate, betas, eps, weight_decay):
    
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    model_uv = UV_Model()

    if model_uv in os.listdir(models_dir):
        model_uv.load_state_dict(torch.load(f'{models_dir}/model_uv'))

    # model_uv.load_state_dict(torch.load(f'{models_dir}/model_uv'))
    
    optimizer = torch.optim.Adam(model_uv.parameters(), lr=learning_rate, betas=betas, eps=1e-8,
                           weight_decay=weight_decay)

    EPOCHS = 2

    print("Training UV model")

    for epoch in range(0,EPOCHS):
        print(f"EPOCH: {epoch}")

        model_uv.train(True)
        avg_loss = train_single_models_epoch(model_uv, epoch, 
            train_dataloader, L2_Dist_Func_Intensity, optimizer)
        
        model_uv.train(False)

        running_vloss = 0

        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            voutputs = model_uv(vinputs)
            vloss = L2_Dist_Func_Intensity(voutputs, vlabels)
            running_vloss += vloss
        
        avg_vloss = running_vloss / (i+1)

        print(f"Loss train {avg_loss} valid {avg_vloss}")

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'{models_dir}/model_uv'
            torch.save(model_uv.state_dict(), model_path)

if __name__ == '__main__':
    splits = {'train':0.7, 'validate':0.1, 'test':0.2}
    training_set, validation_set, test_set = load_datasets(splits)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=6)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=6)

    train_single_models(training_loader, validation_loader, 1e-3, (0.9, 0.999), 1e-8, 1e-4)