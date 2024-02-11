from accelerate.logging import get_logger
from accelerate import Accelerator
from dataset_configuration import prepare_dataset,Disparity_Normalization,resize_max_res_tensor

import matplotlib.pyplot as plt
import torch

# args
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/mnt/disks/data1'
dataset_name='sceneflow'
dataset_path='/mnt/disks/data1'
trainlist='../datafiles/sceneflow/SceneFlow_With_Occ.list'
vallist='../datafiles/sceneflow/FlyingThings3D_Test_With_Occ.list'
output_dir='../outputs/sceneflow_fine_tune_hardest'
train_batch_size=2
num_train_epochs=10
gradient_accumulation_steps=8
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='sceneflow_pretrain_tracker_hardest'
checkpointing_steps=1000

Accelerator()
logger = get_logger(__name__, log_level="INFO")


(train_loader,test_loader), dataset_config_dict = prepare_dataset(
    data_name=dataset_name,
    datapath=dataset_path,
    trainlist=trainlist,
    vallist=vallist,
    batch_size=train_batch_size,
    test_batch=1,
    datathread=dataloader_num_workers,
    logger=logger)

print("num batches", len(train_loader))

for step, batch in enumerate(train_loader):
    # convert the images and the depths into lantent space.
    image = batch['img_left'] # Shape = [batchsize, channels=3, v=540, h=960]
    disparity = batch['gt_disp'] # Shape = [batchsize, v=540, h=960]

    print("image range:", torch.min(image), torch.max(image)) # [0,1]
    print("mask range:", torch.min(disparity), torch.max(disparity))    # [0,127]
    # plt.imshow(  di[0].permute(1, 2, 0)  )
    # plt.show()
                
    # disparity is only a single channel so copy it across 3
    disp_single = disparity.unsqueeze(0)
    disparity_stacked = disp_single.repeat(1,3,1,1) # dim 0 is batch?

    print("image type:", image.dtype) # [-1,1]
    print("mask type:", disparity.dtype)    # [0,1]

    image_resized = resize_max_res_tensor(image,is_disp=False) #range in (0-1)
    disparity_resized = resize_max_res_tensor(disparity_stacked,is_disp=True) # not range



    # depth normalization: [([1, 3, 432, 768])]
    disparity_resized_normalized = Disparity_Normalization(disparity_resized)

    break