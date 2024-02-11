LAUNCH_TRAINING(){

# accelerate config default
cd .. 
cd training
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/mnt/disks/data1'
dataset_name='trans10k'
trainlist='../datafiles/sceneflow/SceneFlow_With_Occ.list'
vallist='../datafiles/sceneflow/FlyingThings3D_Test_With_Occ.list'
output_dir='../outputs/sceneflow_fine_tune_hardest'
train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=8
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='sceneflow_pretrain_tracker_hardest'
checkpointing_steps=1000

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"  --multi_gpu depth2image_trainer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_name  $dataset_name --trainlist $trainlist \
                  --dataset_path $root_path --vallist $vallist \
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing \
                  --checkpointing_steps $checkpointing_steps \
                #   --enable_xformers_memory_efficient_attention \

}



LAUNCH_TRAINING
