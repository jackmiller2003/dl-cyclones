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
import pickle

data_dir = '/g/data/x77/ob2720/cyclone_binaries/'
models_dir = '/g/data/x77/jm0124/models'

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12350'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_single_models(train_dataset_uv, val_dataset_uv, train_dataset_z, val_dataset_z, train_dataset_meta, val_dataset_meta, learning_rate, betas, eps, weight_decay):
    
    print("-------- Loading models --------")

    model_uv = UV_Model()
    model_z = Z_Model()
    model_meta = Meta_Model()

    if False:#'model_uv-0.0038049714482137156' in os.listdir(models_dir):
        EPOCHS = 10

        if True:
            state = torch.load(f'{models_dir}/model_uv-64.7119321766501')            
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

        optimizer_params = {
            'learning_rate':1e-3,
            'betas':betas,
            'eps':eps,
            'weight_decay':weight_decay,
            'optimizer_dict':{}
        }
        
        print("Spawned processes")
        
        for epoch in range(1,EPOCHS+1):
        
            world_size = 2
            mp.spawn(
                train,
                args=(world_size, train_dataset_uv, model_uv, optimizer_params, epoch, "model_uv"),
                nprocs=world_size
            )
            
            state = torch.load(f'{models_dir}/model_uv_scratch')
            
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
            
            optimizer_params = {
                'learning_rate':1e-3,
                'betas':betas,
                'eps':eps,
                'weight_decay':weight_decay,
                'optimizer_dict':optimizer_dict
            }

            mp.spawn(
                validate,
                args=(world_size, val_dataset_uv, model_uv, optimizer_params, epoch, "model_uv"),
                nprocs=world_size
            )

    if False:
        EPOCHS = 10
        print("model_z")
        if False:
            state = torch.load(f'{models_dir}/model_z-130.15425896357115')

            state_dict = state['state_dict']
            optimizer_dict = state['optimizer']

            model_dict = OrderedDict()
            pattern = re.compile('module.')
            for k,v in state_dict.items():
                if re.search("module", k):
                    model_dict[re.sub(pattern, '', k)] = v
                else:
                    model_dict = state_dict
            model_z.load_state_dict(model_dict)

        optimizer_params = {
            'learning_rate':1e-5,
            'betas':betas,
            'eps':eps,
            'weight_decay':weight_decay,
            'optimizer_dict':{}
        }

        print("Spawned processes")

        for epoch in range(1,EPOCHS+1):

            world_size = 2
            mp.spawn(
                train,
                args=(world_size, train_dataset_z, model_z, optimizer_params, epoch, "model_z"),
                nprocs=world_size
            )
            
            print("Up to here...")
            
            state = torch.load(f'{models_dir}/model_z_scratch')
            
            state_dict = state['state_dict']
            optimizer_dict = state['optimizer']

            model_dict = OrderedDict()
            pattern = re.compile('module.')
            for k,v in state_dict.items():
                if re.search("module", k):
                    model_dict[re.sub(pattern, '', k)] = v
                else:
                    model_dict = state_dict
            model_z.load_state_dict(model_dict)

            # This needs to be changed to not call save because it will call it twice.
            mp.spawn(
                validate,
                args=(world_size, val_dataset_z, model_z, optimizer_params, epoch, "model_z"),
                nprocs=world_size
            )
        
    if True: #'model_meta-0.004831329930796832' in os.listdir(models_dir):
        EPOCHS = 35
        print("model_meta")
#         state = torch.load(f'{models_dir}/model_meta-0.004831329930796832')

#         state_dict = state['state_dict']
#         optimizer_dict = state['optimizer']

#         model_dict = OrderedDict()
#         pattern = re.compile('module.')
#         for k,v in state_dict.items():
#             if re.search("module", k):
#                 model_dict[re.sub(pattern, '', k)] = v
#             else:
#                 model_dict = state_dict
#         model_meta.load_state_dict(model_dict)

        optimizer_params = {
            'learning_rate':2e-3,
            'betas':betas,
            'eps':eps,
            'weight_decay':weight_decay,
            'optimizer_dict':{}
        }

        print("Spawned processes")

        for epoch in range(1,EPOCHS+1):

            world_size = 2
            mp.spawn(
                train,
                args=(world_size, train_dataset_meta, model_meta, optimizer_params, epoch, "model_meta"),
                nprocs=world_size
            )
            
            print("Up to here...")
            
            state = torch.load(f'{models_dir}/model_meta_scratch')
            
            state_dict = state['state_dict']
            optimizer_dict = state['optimizer']

            model_dict = OrderedDict()
            pattern = re.compile('module.')
            for k,v in state_dict.items():
                if re.search("module", k):
                    model_dict[re.sub(pattern, '', k)] = v
                else:
                    model_dict = state_dict
            model_meta.load_state_dict(model_dict)

            # This needs to be changed to not call save because it will call it twice.
            mp.spawn(
                validate,
                args=(world_size, val_dataset_meta, model_meta, optimizer_params, epoch, "model_meta"),
                nprocs=world_size
            )
        
        return
        

def train_fusion_model(train_concat_ds, val_concat_ds, learning_rate, betas, eps, weight_decay, reimport=False):
    
    model_fusion = Fusion_Model()

    if ('model_fusion' in os.listdir(models_dir)) and reimport:
        model_fusion.load_state_dict(torch.load(f'{models_dir}/model_fusion'))
    else:
        print("-------- Loading models --------")

        model_uv = UV_Model()

        # Need to specify here the exact models
        if 'model_uv-64.97959783643554' in os.listdir(models_dir):

            state = torch.load(f'{models_dir}/model_uv-64.97959783643554')

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
        
        print("Model UV loaded")
        
        model_z = Z_Model()

        if 'model_z-180.81762405105079' in os.listdir(models_dir):
            state = torch.load(f'{models_dir}/model_z-180.81762405105079')

            state_dict = state['state_dict']
            optimizer_dict = state['optimizer']

            model_dict = OrderedDict()
            pattern = re.compile('module.')
            for k,v in state_dict.items():
                if re.search("module", k):
                    model_dict[re.sub(pattern, '', k)] = v
                else:
                    model_dict = state_dict
                    
            model_z.load_state_dict(model_dict)
        
        print("Model z loaded")

        model_meta = Meta_Model()

        if 'model_meta-389.45942573206014' in os.listdir(models_dir):
            state = torch.load(f'{models_dir}/model_meta-389.45942573206014')

            state_dict = state['state_dict']
            optimizer_dict = state['optimizer']

            model_dict = OrderedDict()
            pattern = re.compile('module.')
            for k,v in state_dict.items():
                if re.search("module", k):
                    model_dict[re.sub(pattern, '', k)] = v
                else:
                    model_dict = state_dict
                    
            model_meta.load_state_dict(model_dict)
        
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
        
        optimizer_dict = {}

        optimizer_params = {
            'learning_rate':1e-3,
            'betas':betas,
            'eps':eps,
            'weight_decay':weight_decay,
            'optimizer_dict':optimizer_dict
        }

        print("Spawned processes")
        
        EPOCHS = 10

        for epoch in range(1,EPOCHS+1):

            world_size = 2
            mp.spawn(
                train,
                args=(world_size, train_concat_ds, model_fusion, optimizer_params, epoch, "model_fusion"),
                nprocs=world_size
            )
            
            print("Up to here...")
            
            state = torch.load(f'{models_dir}/model_fusion_scratch')
            
            state_dict = state['state_dict']
            optimizer_dict = state['optimizer']

            model_dict = OrderedDict()
            pattern = re.compile('module.')
            for k,v in state_dict.items():
                if re.search("module", k):
                    model_dict[re.sub(pattern, '', k)] = v
                else:
                    model_dict = state_dict
            model_fusion.load_state_dict(model_dict)

            optimizer_params = {
                'learning_rate':1e-3,
                'betas':betas,
                'eps':eps,
                'weight_decay':weight_decay,
                'optimizer_dict':optimizer_dict
            }

            # This needs to be changed to not call save because it will call it twice.
            mp.spawn(
                validate,
                args=(world_size, val_concat_ds, model_fusion, optimizer_params, epoch, "model_fusion"),
                nprocs=world_size
            )

def prepare(rank, world_size, dataset, batch_size=64, pin_memory=False, num_workers=8):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, sampler=sampler)
    
    print(f"Length of dataloader is {len(dataloader)}")
    
    return dataloader


def train(rank, world_size, dataset, model, optimizer_params, epoch, model_name):
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
    
    learning_rate = optimizer_params["learning_rate"]
    betas = optimizer_params["betas"]
    eps = optimizer_params["eps"]
    weight_decay = optimizer_params["weight_decay"]
    optimizer_dict = optimizer_params["optimizer_dict"]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                        weight_decay=1e-4)
    
    # optimizer.load_state_dict(optimizer_dict)
    
    # if we are using DistributedSampler, we have to tell it which epoch this is
    dataloader.sampler.set_epoch(epoch)
    
    loss_fn = L2_Dist_Func_Intensity().to(rank)
    
    model.train()

    with torch.autograd.profiler.profile() as prof:
    
        for step, (example, target) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            pred = model(example).to(rank)
            label = target.to(rank)

            loss = loss_fn(pred, label, rank)

            loss = loss

            loss.backward()

            optimizer.step()

            if step % 10 == 9:
                print(f"{loss} loss for step {step} in {epoch}")
    
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    
    model_path = f'{models_dir}/{model_name}_scratch'
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    print("Saved model")
    torch.save(state, model_path)
    
    cleanup()

def validate(rank, world_size, dataset, model, optimizer_params, epoch, model_name):
    setup(rank, world_size)
    # prepare the dataloader
    dataloader = prepare(rank, world_size, dataset)
    
    model = model.to(rank)
    
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # if we are using DistributedSampler, we have to tell it which epoch this is
    dataloader.sampler.set_epoch(epoch)
    
    loss_fn = L2_Dist_Func_Intensity().to(rank)
    
    running_vloss = 0

    best_vloss = 1e10
    
    model.eval()
    
    with torch.no_grad():
        for step, vdata in enumerate(dataloader):
            vinputs, vlabels = vdata
            vinputs = vinputs
            voutputs = model(vinputs).to(rank)
            vlabels = vlabels.to(rank)
            vloss = loss_fn(voutputs, vlabels, rank)
            running_vloss += vloss.mean().item()

            if step % 10 == 9:
                print(f"{vloss} loss for step {step} in {epoch}")
    
    avg_vloss = running_vloss / (step+1)

    print(f"Loss train for {model_name} valid {avg_vloss}")

    for saved_model in os.listdir(models_dir):
        if f"{model_name}-" in saved_model:
            other_vloss = float(saved_model.split(f"{model_name}-",1)[1])
            if other_vloss < best_vloss:
                best_vloss = other_vloss
    
    print(f"Best vloss is {best_vloss}")
    
    if avg_vloss < best_vloss and rank == 0:
        best_vloss = avg_vloss
        model_path = f'{models_dir}/{model_name}-{str(best_vloss)}'
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer_params["optimizer_dict"]
        }
        print("Saved model")
        torch.save(state, model_path)
        
    cleanup()

def eval(rank, world_size, dataset, model, model_name):
    epoch = 1
    setup(rank, world_size)
    # prepare the dataloader
    dataloader = prepare(rank, world_size, dataset)
    
    model = model.to(rank)
    
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # if we are using DistributedSampler, we have to tell it which epoch this is
    dataloader.sampler.set_epoch(epoch)
    
    loss_fn = L2_Dist_Func_Intensity().to(rank)
    
    running_tloss = 0
    
    for step, vdata in enumerate(dataloader):
        tinputs, tlabels = vdata
        toutputs = model(tinputs).to(rank)
        tlabels = tlabels.to(rank)
        tloss = loss_fn(toutputs, tlabels, rank)
        print(f"test labels: {tlabels}")
        print(f"test labels: {toutputs}")
        print(f"test labels: {tloss.mean().item()}")
        running_tloss += tloss.mean().item()

        if step % 10 == 9:
            print(f"{tloss} loss for step {step} in {epoch}")
    
    avg_tloss = running_tloss / (step+1)

    print(f"Loss in testing for {model_name} was {avg_tloss}")
    tloss_and_rank = (avg_tloss, rank)

    with open(f'tloss-{rank}.pickle', 'wb') as f:
        pickle.dump(tloss_and_rank, f)

    cleanup()

def test_model(test_dataset, model_name):
    if model_name == "model_uv":
        state = torch.load(f'{models_dir}/model_uv_scratch')
        
        model_uv = UV_Model()
            
        state_dict = state['state_dict']

        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = state_dict
        model_uv.load_state_dict(model_dict)

        world_size = 2

        mp.spawn(
                eval,
                args=(world_size, test_dataset, model_uv, "model_uv"),
                nprocs=world_size
            )
        
        avg_tloss = 0

        for file in os.listdir():
            if file.endswith('.pickle'):
                with open(file, 'wb') as tloss_tuple:
                    avg_tloss += pickle.load(tloss_tuple)[0]
        
        return (avg_tloss/world_size)


if __name__ == '__main__':
    train_dataset_uv, validate_dataset_uv, test_dataset_uv, train_dataset_z, validate_dataset_z, test_dataset_z, train_dataset_meta, validate_dataset_meta, \
    test_dataset_meta, train_concat_ds, validate_concat_ds, test_concat_ds = load_datasets()            
    
    print("Training fusion model")
    #train_single_models(train_dataset_uv, validate_dataset_uv, train_dataset_z, validate_dataset_z, train_dataset_meta, validate_dataset_meta, 1e-3, (0.9, 0.999), 1e-8, 1e-4)
    train_fusion_model(train_concat_ds, validate_concat_ds, 1e-3, (0.9, 0.999), 1e-8, 1e-4, False)

    # Want to test 2014253N13260