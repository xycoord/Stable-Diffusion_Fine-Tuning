import matplotlib.pyplot as plt
import torch

from dataset_configuration import prepare_dataset,Disparity_Normalization,resize_max_res_tensor

# ======== Trans10k Loader ========

from torchvision import transforms
import torch.utils.data as data

import sys
sys.path.append("..")

from dataloader_trans10k import get_segmentation_dataset
from dataloader_trans10k.settings import cfg
from dataloader_trans10k.distributed import *

# image transform
input_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
])
# dataset and dataloader
data_kwargs = {'transform': input_transform, 'base_size': cfg.TRAIN.BASE_SIZE,
                'crop_size': cfg.TRAIN.CROP_SIZE}
train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **data_kwargs)

iters_per_epoch = len(train_dataset) // (cfg.TRAIN.BATCH_SIZE)
max_iters = cfg.TRAIN.EPOCHS * iters_per_epoch

train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=False)
train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, max_iters, drop_last=True)

train_loader = data.DataLoader( dataset=train_dataset,
                                batch_sampler=train_batch_sampler,
                                num_workers=cfg.DATASET.WORKERS,
                                pin_memory=True)

print("num batches", len(train_loader))

for (image,masks,filename) in train_loader:
    print("images:",image.shape)   # [batchsize, 3, 512, 512]
    print("masks:",masks.shape) # [batchsize, 512, 512]
    print("filename:",filename)

    print("image range:", torch.min(image), torch.max(image)) # [-1,1]
    print("mask range:", torch.min(masks), torch.max(masks))    # [0,1]

    # disparity is only a single channel so copy it across 3
    mask_single = masks.unsqueeze(0)
    masks_stacked = masks.repeat(1,3,1,1) # dim 0 is batch?
    masks_stacked = masks_stacked.float()

    image_resized = resize_max_res_tensor(image,is_disp=False) #range in (0-1)
    print("New Shape Image", image_resized.shape)
    # plt.imshow(image_resized[0].permute(1, 2, 0))
    # plt.show()

    print("image type:", image.dtype) # [-1,1]
    print("mask type:", masks_stacked.dtype)    # [0,1]
    
    masks_resized = resize_max_res_tensor(masks_stacked, is_disp=True) # not range
    print("New Shape Mask", masks_resized.shape)

    masks_resized_resized_normalized = Disparity_Normalization(masks_resized)
    print("mask range:", torch.min(masks_resized_resized_normalized), torch.max(masks_resized_resized_normalized))    # [0,1]



    break