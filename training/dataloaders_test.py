from accelerate.logging import get_logger
from accelerate import Accelerator
from dataset_configuration import prepare_dataset,Disparity_Normalization,resize_max_res_tensor

import matplotlib.pyplot as plt

# # args
# pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
# root_path='/mnt/disks/data1'
# dataset_name='sceneflow'
# dataset_path='/mnt/disks/data1'
# trainlist='../datafiles/sceneflow/SceneFlow_With_Occ.list'
# vallist='../datafiles/sceneflow/FlyingThings3D_Test_With_Occ.list'
# output_dir='../outputs/sceneflow_fine_tune_hardest'
# train_batch_size=2
# num_train_epochs=10
# gradient_accumulation_steps=8
# learning_rate=1e-5
# lr_warmup_steps=0
# dataloader_num_workers=4
# tracker_project_name='sceneflow_pretrain_tracker_hardest'
# checkpointing_steps=1000

# Accelerator()
# logger = get_logger(__name__, log_level="INFO")


# (train_loader,test_loader), dataset_config_dict = prepare_dataset(
#     data_name=dataset_name,
#     datapath=dataset_path,
#     trainlist=trainlist,
#     vallist=vallist,
#     batch_size=train_batch_size,
#     test_batch=1,
#     datathread=dataloader_num_workers,
#     logger=logger)

# print("hi")

# for step, batch in enumerate(train_loader):
#     # convert the images and the depths into lantent space.
#     left_image_data = batch['img_left'] # Shape = [batchsize, channels=3, v=540, h=960]
#     left_disparity = batch['gt_disp'] # Shape = [batchsize, v=540, h=960]
                
#     # disparity is only a single channel so copy it across 3
#     left_disp_single = left_disparity.unsqueeze(0)
#     left_disparity_stacked = left_disp_single.repeat(1,3,1,1) # dim 0 is batch?

#     left_image_data_resized = resize_max_res_tensor(left_image_data,is_disp=False) #range in (0-1)
#     left_disparity_resized = resize_max_res_tensor(left_disparity_stacked,is_disp=True) # not range

#     # depth normalization: [([1, 3, 432, 768])]
#     left_disparity_resized_normalized = Disparity_Normalization(left_disparity_resized)

#     break

# print(left_disparity.shape)
# plt.imshow(  left_image_data[0].permute(1, 2, 0)  )
# plt.show()

# ======== Trans10k Loader ========

from torchvision import transforms
import torch.utils.data as data

from dataloader_trans10k import get_segmentation_dataset
from dataloader_trans10k.settings import cfg
from dataloader_trans10k.distributed import *

# image transform
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
])
# dataset and dataloader
data_kwargs = {'transform': input_transform, 'base_size': cfg.TRAIN.BASE_SIZE,
                'crop_size': cfg.TRAIN.CROP_SIZE}
train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **data_kwargs)
# #debug code
# import cv2
# for i in range(10):
#     img, mask, _ = train_dataset[i]
#     print(img.shape, mask.shape)
#     mask = mask.data.cpu().numpy()*127
#     # mask = cv2.resize(mask, (500,500))
#     # cv2.imwrite('./trash/{}.jpg'.format(i), mask)
# embed(header='check loader')
iters_per_epoch = len(train_dataset) // (cfg.TRAIN.BATCH_SIZE)
max_iters = cfg.TRAIN.EPOCHS * iters_per_epoch

train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=False)
train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, max_iters, drop_last=True)

train_loader2 = data.DataLoader(dataset=train_dataset,
                                    batch_sampler=train_batch_sampler,
                                    num_workers=cfg.DATASET.WORKERS,
                                    pin_memory=True)

for (images,targets,filename) in train_loader2:
    print("images:",images.shape)
    print("targets:",targets.shape)
    print("filename:",filename)
    plt.imshow(targets.permute(1, 2, 0)  )
    plt.show()
    break