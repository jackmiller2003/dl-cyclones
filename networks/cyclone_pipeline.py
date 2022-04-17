from utils.loss_functions import L2_Dist_Func_Intensity
from utils.data_loader import *
import torch
from datetime import datetime
from models.uv_model import UV_Model
from models.z_model import Z_Model
from models.meta_model import Meta_Model
from models.fusion_model import Fusion_Model
import os
import datetime
import xarray
import matplotlib.pyplot as plt
from utils.util_funcs import *
from tqdm import tqdm
from collections import OrderedDict
import re

data_dir = '/g/data/x77/ob2720/cyclone_binaries/'
models_dir = '/g/data/x77/jm0124/models'

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
def prepare(rank, world_size, batch_size=256, pin_memory=True, num_workers=4):
    dataset = Your_Dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

def cleanup():
    dist.destroy_process_group()

def train_single_models_epoch(model, epoch, train_dataloader, loss_func, optimizer, model_name):
    
    running_loss = 0
    last_loss = 0

    torch.autograd.set_detect_anomaly(True)

    mean_loss = []
    
    parallel_net = torch.nn.DataParallel(model, device_ids = [0,1])
    parallel_net = parallel_net.to(0)

    for i, (example, target) in enumerate(train_dataloader):

        example = example.to(0)
        
        optimizer.zero_grad()

        # print(f"Example contains nan {torch.isnan(example).any()}")

        output = parallel_net(example)

        loss = loss_func(output, target)
        loss.mean().backward()

        optimizer.step()

        running_loss += loss.mean().item()

        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            mean_loss.append(last_loss)
            print('batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0
        
        if i % 20 == 19:
            break
        
    print(f"Reached end of train single models for {model_name}")

    plt.plot(list(range(len(mean_loss))), mean_loss)
    plt.savefig(f'Mean-loss {model_name}.png')
    
    return last_loss

def train_component(model, train_dataloader, val_dataloader, loss_fn, optimizer, model_name, num_epochs):
    print(f"-------- Training {model_name} model --------")

    best_vloss = 1e10
    
    for epoch in range(0,num_epochs):
        print(f"EPOCH: {epoch}")

        model.train(True)
        avg_loss = train_single_models_epoch(model, epoch, 
            train_dataloader, loss_fn, optimizer, model_name)

        parallel_net = torch.nn.DataParallel(model, device_ids = [0,1])
        parallel_net = parallel_net.to(0)
        
        running_vloss = 0

        for i, vdata in tqdm(enumerate(val_dataloader)):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(0)
            voutputs = parallel_net(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.mean().item()
            
            if i % 20 == 19:
                break
        
        avg_vloss = running_vloss / (i+1)

        print(f"Loss train for {model_name} {avg_loss} valid {avg_vloss}")

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'{models_dir}/{model_name}-{str(best_vloss)}'
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, model_path)

    print(f"Best vloss for {model_name} was {best_vloss}")

def train_single_models(train_dataset_uv, val_dataset_uv, learning_rate, betas, eps, weight_decay):
    
    print("-------- Loading models --------")

    model_uv = UV_Model()
    
    EPOCHS = 1

    if 'model_uv-0.0042708070717918205' in os.listdir(models_dir):
        print("model_uv-0.0042708070717918205")
        state = torch.load(f'{models_dir}/model_uv-0.0042708070717918205')
        
        state_dict = state['state_dict']
        optimizer_dict = state['optimizer']
        
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = state_dict
        model_uv.load_state_dict(model_dict)
        
        optimizer = torch.optim.Adam(model_uv.parameters(), lr=learning_rate, betas=betas, eps=1e-8,
                           weight_decay=weight_decay)
        
        optimizer.load_state_dict(optimizer_dict)
        
        world_size = 2
        mp.spawn(
            train,
            args=(world_size, train_dataset_uv, model_uv, optimizer, 1),
            nprocs=world_size
        )
                
        return
        
        
    # print("Model UV loaded")
    
    # model_z = Z_Model()

    # if 'model_z' in os.listdir(models_dir):
    #     model_z.load_state_dict(torch.load(f'{models_dir}/model_z'))
    
    # print("Model z loaded")

    # model_meta = Meta_Model()

    # if 'model_z' in os.listdir(models_dir):
    #     model_meta.load_state_dict(torch.load(f'{models_dir}/model_meta'))
    
    # print("Model meta loaded")

    # Here we are using the same optimizer, but we could simply change the learning rate
    # in order to accomodate the differences needed for different networks.

#     optimizer = torch.optim.Adam(model_uv.parameters(), lr=learning_rate, betas=betas, eps=1e-8,
#                            weight_decay=weight_decay)

#     train_component(model_uv, train_dataloader_uv, val_dataloader_uv, loss_fn, optimizer, "model_uv", EPOCHS)

#     optimizer = torch.optim.Adam(model_z.parameters(), lr=learning_rate, betas=betas, eps=1e-8,
#                         weight_decay=weight_decay)
    
#     train_component(model_z, train_dataloader_z, val_dataloader_z, loss_fn, optimizer, "model_z", EPOCHS)

    optimizer = torch.optim.Adam(model_meta.parameters(), lr=learning_rate*0.01, betas=betas, eps=1e-8,
                        weight_decay=weight_decay)

    train_component(model_meta, train_dataloader_meta, val_dataloader_meta, loss_fn, optimizer, "model_meta", EPOCHS)

def train_fusion_model(train_concat_ds, val_concat_ds, learning_rate, betas, eps, weight_decay, reimport=False):
    
    model_fusion = Fusion_Model()
    loss_func = torch.nn.DataParallel(L2_Dist_Func_Intensity(), device_ids=[0,1])
    loss_func.to(0)

    if ('model_fusion' in os.listdir(models_dir)) and not reimport:
        model_fusion.load_state_dict(torch.load(f'{models_dir}/model_fusion'))
    else:
        print("-------- Loading models --------")

        model_uv = UV_Model()

        if 'model_uv' in os.listdir(models_dir):
            model_uv.load_state_dict(torch.load(f'{models_dir}/model_uv'))
        
        print("Model UV loaded")
        
        model_z = Z_Model()

        if 'model_z' in os.listdir(models_dir):
            model_z.load_state_dict(torch.load(f'{models_dir}/model_z'))
        
        print("Model z loaded")

        model_meta = Meta_Model()

        if 'model_z' in os.listdir(models_dir):
            model_meta.load_state_dict(torch.load(f'{models_dir}/model_meta'))
        
        print("Model meta loaded")

        model_fusion = Fusion_Model()

        pretrained_dict_uv = model_uv.state_dict()
        pretrained_dict_z = model_z.state_dict()
        pretrained_dict_meta = model_meta.state_dict()
        model_fusion_dict = model_fusion.state_dict()

        # Imports for UV model

        model_fusion_dict['conv1_uv.weight'] = pretrained_dict_uv['conv1.weight']
        model_fusion_dict['conv1_uv.bias'] = pretrained_dict_uv['conv1.bias']
        model_fusion_dict['conv1_bn_uv.weight'] = pretrained_dict_uv['conv1_bn.weight']
        model_fusion_dict['conv1_bn_uv.bias'] = pretrained_dict_uv['conv1_bn.bias']
        model_fusion_dict['conv1_bn_uv.running_mean'] = pretrained_dict_uv['conv1_bn.running_mean']
        model_fusion_dict['conv1_bn_uv.running_var'] = pretrained_dict_uv['conv1_bn.running_var']

        model_fusion_dict['conv2_uv.weight'] = pretrained_dict_uv['conv2.weight']
        model_fusion_dict['conv2_uv.bias'] = pretrained_dict_uv['conv2.bias']
        model_fusion_dict['conv2_bn_uv.weight'] = pretrained_dict_uv['conv2_bn.weight']
        model_fusion_dict['conv2_bn_uv.bias'] = pretrained_dict_uv['conv2_bn.bias']
        model_fusion_dict['conv2_bn_uv.running_mean'] = pretrained_dict_uv['conv2_bn.running_mean']
        model_fusion_dict['conv2_bn_uv.running_var'] = pretrained_dict_uv['conv2_bn.running_var']

        model_fusion_dict['conv3_uv.weight'] = pretrained_dict_uv['conv3.weight']
        model_fusion_dict['conv3_uv.bias'] = pretrained_dict_uv['conv3.bias']
        model_fusion_dict['conv3_bn_uv.weight'] = pretrained_dict_uv['conv3_bn.weight']
        model_fusion_dict['conv3_bn_uv.bias'] = pretrained_dict_uv['conv3_bn.bias']
        model_fusion_dict['conv3_bn_uv.running_mean'] = pretrained_dict_uv['conv3_bn.running_mean']
        model_fusion_dict['conv3_bn_uv.running_var'] = pretrained_dict_uv['conv3_bn.running_var']

        model_fusion_dict['fc1_uv.weight'] = pretrained_dict_uv['fc1.weight']
        model_fusion_dict['fc1_uv.bias'] = pretrained_dict_uv['fc1.bias']
        model_fusion_dict['fc1_bn_uv.weight'] = pretrained_dict_uv['fc1_bn.weight']
        model_fusion_dict['fc1_bn_uv.bias'] = pretrained_dict_uv['fc1_bn.bias']
        model_fusion_dict['fc1_bn_uv.running_mean'] = pretrained_dict_uv['fc1_bn.running_mean']
        model_fusion_dict['fc1_bn_uv.running_var'] = pretrained_dict_uv['fc1_bn.running_var']

        model_fusion_dict['fc2_uv.weight'] = pretrained_dict_uv['fc2.weight']
        model_fusion_dict['fc2_uv.bias'] = pretrained_dict_uv['fc2.bias']
        model_fusion_dict['fc2_bn_uv.weight'] = pretrained_dict_uv['fc2_bn.weight']
        model_fusion_dict['fc2_bn_uv.bias'] = pretrained_dict_uv['fc2_bn.bias']
        model_fusion_dict['fc2_bn_uv.running_mean'] = pretrained_dict_uv['fc2_bn.running_mean']
        model_fusion_dict['fc2_bn_uv.running_var'] = pretrained_dict_uv['fc2_bn.running_var']

        model_fusion_dict['fc3_uv.weight'] = pretrained_dict_uv['fc3.weight']
        model_fusion_dict['fc3_uv.bias'] = pretrained_dict_uv['fc3.bias']
        model_fusion_dict['fc3_bn_uv.weight'] = pretrained_dict_uv['fc3_bn.weight']
        model_fusion_dict['fc3_bn_uv.bias'] = pretrained_dict_uv['fc3_bn.bias']
        model_fusion_dict['fc3_bn_uv.running_mean'] = pretrained_dict_uv['fc3_bn.running_mean']
        model_fusion_dict['fc3_bn_uv.running_var'] = pretrained_dict_uv['fc3_bn.running_var']

        # Imports for Z model

        model_fusion_dict['conv1_z.weight'] = pretrained_dict_z['conv1.weight']
        model_fusion_dict['conv1_z.bias'] = pretrained_dict_z['conv1.bias']
        model_fusion_dict['conv1_bn_z.weight'] = pretrained_dict_z['conv1_bn.weight']
        model_fusion_dict['conv1_bn_z.bias'] = pretrained_dict_z['conv1_bn.bias']
        model_fusion_dict['conv1_bn_z.running_mean'] = pretrained_dict_z['conv1_bn.running_mean']
        model_fusion_dict['conv1_bn_z.running_var'] = pretrained_dict_z['conv1_bn.running_var']

        model_fusion_dict['conv2_z.weight'] = pretrained_dict_z['conv2.weight']
        model_fusion_dict['conv2_z.bias'] = pretrained_dict_z['conv2.bias']
        model_fusion_dict['conv2_bn_z.weight'] = pretrained_dict_z['conv2_bn.weight']
        model_fusion_dict['conv2_bn_z.bias'] = pretrained_dict_z['conv2_bn.bias']
        model_fusion_dict['conv2_bn_z.running_mean'] = pretrained_dict_z['conv2_bn.running_mean']
        model_fusion_dict['conv2_bn_z.running_var'] = pretrained_dict_z['conv2_bn.running_var']

        model_fusion_dict['conv3_z.weight'] = pretrained_dict_z['conv3.weight']
        model_fusion_dict['conv3_z.bias'] = pretrained_dict_z['conv3.bias']
        model_fusion_dict['conv3_bn_z.weight'] = pretrained_dict_z['conv3_bn.weight']
        model_fusion_dict['conv3_bn_z.bias'] = pretrained_dict_z['conv3_bn.bias']
        model_fusion_dict['conv3_bn_z.running_mean'] = pretrained_dict_z['conv3_bn.running_mean']
        model_fusion_dict['conv3_bn_z.running_var'] = pretrained_dict_z['conv3_bn.running_var']

        model_fusion_dict['fc1_z.weight'] = pretrained_dict_z['fc1.weight']
        model_fusion_dict['fc1_z.bias'] = pretrained_dict_z['fc1.bias']
        model_fusion_dict['fc1_bn_z.weight'] = pretrained_dict_z['fc1_bn.weight']
        model_fusion_dict['fc1_bn_z.bias'] = pretrained_dict_z['fc1_bn.bias']
        model_fusion_dict['fc1_bn_z.running_mean'] = pretrained_dict_z['fc1_bn.running_mean']
        model_fusion_dict['fc1_bn_z.running_var'] = pretrained_dict_z['fc1_bn.running_var']

        model_fusion_dict['fc2_z.weight'] = pretrained_dict_z['fc2.weight']
        model_fusion_dict['fc2_z.bias'] = pretrained_dict_z['fc2.bias']
        model_fusion_dict['fc2_bn_z.weight'] = pretrained_dict_z['fc2_bn.weight']
        model_fusion_dict['fc2_bn_z.bias'] = pretrained_dict_z['fc2_bn.bias']
        model_fusion_dict['fc2_bn_z.running_mean'] = pretrained_dict_z['fc2_bn.running_mean']
        model_fusion_dict['fc2_bn_z.running_var'] = pretrained_dict_z['fc2_bn.running_var']

        model_fusion_dict['fc3_z.weight'] = pretrained_dict_z['fc3.weight']
        model_fusion_dict['fc3_z.bias'] = pretrained_dict_z['fc3.bias']
        model_fusion_dict['fc3_bn_z.weight'] = pretrained_dict_z['fc3_bn.weight']
        model_fusion_dict['fc3_bn_z.bias'] = pretrained_dict_z['fc3_bn.bias']
        model_fusion_dict['fc3_bn_z.running_mean'] = pretrained_dict_z['fc3_bn.running_mean']
        model_fusion_dict['fc3_bn_z.running_var'] = pretrained_dict_z['fc3_bn.running_var']

        # For the meta model

        model_fusion_dict['fc1_meta.weight'] = pretrained_dict_meta['fc1.weight']
        model_fusion_dict['fc1_meta.bias'] = pretrained_dict_meta['fc1.bias']

        """
        Taken directly from Fussion_CNN_hurricanes
        See file: https://github.com/sophiegif/FusionCNN_hurricanes/blob/d2a48a3aa8ac75f5cbc506f7dfc5977e2df3abff/script_launch_fusion.py#L121-L277
        However, removed additional fc layer init.
        """

        # load weigths for fusion model
        model_fusion.load_state_dict(model_fusion_dict)

        # set unfused layers freezed
        num_params = 0
        for param in model_fusion.parameters():
            num_params += 1
        num_unfreezed_params = len(('fc5.weight', 'fc6.weight', 'fc7.weight', 'fc5.bias',
                                    'fc6.bias', 'fc7.bias'))
        for counter, param in enumerate(model_fusion.parameters()):
            # This here is some dodgy code
            if param.size() == model_fusion_dict['fc2_meta.weight'].size() or param.size() == model_fusion_dict['fc2_meta.bias'].size():
                if (not (False in torch.eq(model_fusion_dict['fc2_meta.weight'], param))) or (not (False in torch.eq(model_fusion_dict['fc2_meta.bias'], param))):
                    # print("hit")
                    param.requires_grad = True
            if counter >= num_params-num_unfreezed_params:
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_fusion.parameters()), lr=learning_rate*0.001,
                            betas=betas, eps=eps, weight_decay=weight_decay)
        
        running_loss = 0
        last_loss = 0

        torch.autograd.set_detect_anomaly(True)

        mean_loss = []

        for i, return_list in enumerate(train_concat_ds):

            atm_data = return_list[0][0]
            meta_data = return_list[1][0]

            target = return_list[0][1]

            optimizer.zero_grad()

            # print(f"Example contains nan {torch.isnan(atm_data).any()}")

            output = model_fusion(atm_data, meta_data) # God I hope this works lol

            loss = loss_func(output, target)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                mean_loss.append(last_loss)
                print('batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0
            
            if i == 201:
                break
        
        model_path = f'{models_dir}/model_fusion'
        torch.save(model_fusion.state_dict(), model_path)

        print("Reached end of fusion training")

        plt.plot(list(range(len(mean_loss))), mean_loss)
        plt.savefig('Mean-loss-fusion.png')
        
def model_eval(validation_loader_uvzmeta):
    model_fusion = Fusion_Model()

    if ('model_fusion' in os.listdir(models_dir)):
        model_fusion.load_state_dict(torch.load(f'{models_dir}/model_fusion'))

    mse_list = calculate_average_mse(model_fusion, validation_loader_uvzmeta)

    print(mse_list)

    plt.plot(list(range(len(mse_list))), mse_list)
    plt.savefig('mse.png')

def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, sampler=sampler)
    
    return dataloader


def train(rank, world_size, dataset, model, optimizer, epoch):
    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    dataloader = prepare(rank, world_size, dataset)
    
    # instantiate the model(it's your own model) and move it to the right device
    model = model.to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    # if we are using DistributedSampler, we have to tell it which epoch this is
    dataloader.sampler.set_epoch(epoch)

    for step, (example, target) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        pred = model(example).to(rank)
        label = target.to(rank)
        
        loss_fn = L2_Dist_Func_Intensity().to(rank)

        loss = loss_fn(pred, label, rank)
        loss = loss.to(rank)
        loss.backward()
        optimizer.step()

        if step % 10 == 9:
            print(f"loss for batch {step} in {epoch}")
    cleanup()

def validate(rank, world_size, dataset, model, optimizer, loss_fn, epoch):
    setup(rank, world_size)
    # prepare the dataloader
    dataloader = prepare(rank, world_size, dataset)
    
    model = model.to(rank)
    
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # if we are using DistributedSampler, we have to tell it which epoch this is
    dataloader.sampler.set_epoch(epoch)
    
    for i, vdata in tqdm(enumerate(dataloader)):
        vinputs, vlabels = vdata
        vinputs = vinputs.to(0)
        voutputs = parallel_net(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss.mean().item()

        if i % 20 == 19:
            break

    avg_vloss = running_vloss / (i+1)

    print(f"Loss train for {model_name} {avg_loss} valid {avg_vloss}")

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'{models_dir}/{model_name}-{str(best_vloss)}'
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, model_path)
    
    
if __name__ == '__main__':
    splits = {'train':0.8, 'validate':0.1, 'test':0.1}
    train_dataset_uv, validate_dataset_uv, test_dataset_uv, train_dataset_z, validate_dataset_z, test_dataset_z, train_dataset_meta, validate_dataset_meta, \
    test_dataset_meta, train_concat_ds, validate_concat_ds, test_concat_ds = load_datasets(splits)

    print("Training single models")

    train_single_models(train_dataset_uv, validate_dataset_uv, 1e-3, (0.9, 0.999), 1e-8, 1e-4)
    
    """
    Need to be very careful here about shuffling. In the fusion training we require two dataloaders at the same time. One which indexes
    the other so they can't be shuffled.
    """

    # training_loader_uvzmeta = torch.utils.data.DataLoader(train_concat_ds, batch_size=4, shuffle=False, num_workers=4, drop_last=True)
    # validation_loader_uvzmeta = torch.utils.data.DataLoader(validate_concat_ds, batch_size=4, shuffle=False, num_workers=4, drop_last=True)

    # train_fusion_model(training_loader_uvzmeta, validation_loader_uvzmeta, 1e-3, (0.9, 0.999), 1e-8, 1e-4, False)

    # model_eval(validation_loader_uvzmeta)