import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

from dataloader.sceneflow_loader import StereoDataset
from torch.utils.data import DataLoader
from dataloader import transforms
import os

from torchvision import transforms
import torch.utils.data as data

from dataloader_trans10k import get_segmentation_dataset
from dataloader_trans10k.settings import cfg
from dataloader_trans10k.distributed import *


# Get Dataset Here
def prepare_dataset(data_name,
                    datapath=None,
                    trainlist=None,
                    vallist=None,
                    batch_size=1,
                    test_batch=1,
                    datathread=4,
                    logger=None):
    
    # set the config parameters
    dataset_config_dict = dict()
    
    if data_name == 'sceneflow':
        train_transform_list = [transforms.ToTensor(),]
        train_transform = transforms.Compose(train_transform_list)

        val_transform_list = [transforms.ToTensor(),]
        val_transform = transforms.Compose(val_transform_list)
        
        train_dataset = StereoDataset(data_dir=datapath,train_datalist=trainlist,test_datalist=vallist,
                                dataset_name='SceneFlow',mode='train',transform=train_transform)
        test_dataset = StereoDataset(data_dir=datapath,train_datalist=trainlist,test_datalist=vallist,
                                dataset_name='SceneFlow',mode='val',transform=val_transform)

    elif data_name == 'trans10k':
        train_transform_list = [transforms.ToTensor(),]
        train_transform = transforms.Compose(train_transform_list)

        val_transform_list = [transforms.ToTensor(),]
        val_transform = transforms.Compose(val_transform_list)

        test_transform_list = [transforms.ToTensor(),]
        test_transform = transforms.Compose(test_transform_list)
        

        
        train_kwargs = {'transform': train_transform, 
                        'base_size': cfg.TRAIN.BASE_SIZE,
                        'crop_size': cfg.TRAIN.CROP_SIZE}
        train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **train_kwargs)

        val_kwargs = {'transform': train_transform, 
                      'base_size': cfg.TRAIN.BASE_SIZE,
                      'crop_size': cfg.TRAIN.CROP_SIZE}
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='validation', mode='val', difficulty='mix', **train_kwargs)

        test_kwargs = {'transform': test_transform, 
                       'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}
        test_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='test', mode='testval', difficulty='easy', **test_kwargs)


    img_height, img_width = train_dataset.get_img_size()


    datathread=4
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    
    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    val_loader = DataLoader(val_dataset, batch_size = batch_size, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = test_batch, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    
    num_batches_per_epoch = len(train_loader)
    
    dataset_config_dict['num_batches_per_epoch'] = num_batches_per_epoch
    dataset_config_dict['img_size'] = (img_height,img_width)
    
    
    return (train_loader, val_loader, test_loader),dataset_config_dict

def Disparity_Normalization(mask):
    clipped_mask = torch.clamp(mask, 0, 1)
    # normalized_disparity = ((mask -min_value)/(max_value-min_value+1e-5) - 0.5) * 2    
    normalized_mask = (clipped_mask - 0.5) *2
    return normalized_mask

def resize_max_res_tensor(input_tensor,is_disp=False,recom_resolution=768):
    assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]
    
    downscale_factor = min(recom_resolution/original_H,
                           recom_resolution/original_W)
    
   
    if is_disp:
        resized_input_tensor = F.interpolate(input_tensor,
                                             scale_factor=downscale_factor, 
                                             mode='nearest')

        return resized_input_tensor * downscale_factor
    else:
        resized_input_tensor = F.interpolate(input_tensor,
                                             scale_factor=downscale_factor,
                                             mode='bilinear',
                                             align_corners=False)
 
        return resized_input_tensor