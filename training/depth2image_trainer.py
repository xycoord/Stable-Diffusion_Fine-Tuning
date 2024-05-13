import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision.transforms import v2

import os
import logging
import tqdm

from accelerate import Accelerator
import transformers
import datasets

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import accelerate

from dataset_configuration import prepare_dataset,Disparity_Normalization,resize_max_res_tensor

from args_parsing import parse_args
from log_val import log_validation
from run_inference import run_inference
from training_utils import new_empty_prompt, encode_to_latent_space
from checkpointing import resume_from, save_checkpoint, setup_custom_hooks 
from noise_samplers import annealed_pyramid_noise_like, pyramid_noise_like

import sys

sys.path.append("..")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main():

    ''' ------------------------Configs Preparation----------------------------'''
    # give the args parsers
    args = parse_args()

    # save  the tensorboard log files
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    # TODO: check config 
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )

    # ==== Setup Logging ====

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True) # only show logs on the main process
    
    # set the warning levels
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if args.seed is not None:
        set_seed(args.seed)


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    


    ''' ------------------------Non-NN Modules Definition----------------------------'''

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    # train_noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    # val_noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer')
    logger.info("loading the noise scheduler and the tokenizer from {}".format(args.pretrained_model_name_or_path), main_process_only=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Load Stable Diffusion 2
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder='vae')
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder='text_encoder')
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            # 4 for RGB, 4 for Mask
            in_channels=8, 
            sample_size=96,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True)

    # Freeze vae and text_encoder and set unet to trainable.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train() # only make the unet-trainable
    
    ema_unet = None
    # using EMA (Exponential Moving Average)
    if args.use_ema:
        logger.info("====== USING EMA ======")
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,subfolder="unet",
            in_channels=8, sample_size=96,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config, device = unet.device)

    # using xformers for efficient attention
    if args.enable_xformers_memory_efficient_attention: 
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        setup_custom_hooks(args, accelerator, ema_unet)
        
    # using checkpoint for saving the memories
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # how many cards did we use: accelerator.num_processes
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # ==== Initialize the optimizer ====
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ========= Setup Dataloaders ===========
    with accelerator.main_process_first():
        (train_loader, val_loader, test_loader), dataset_config_dict = prepare_dataset(
            data_name=args.dataset_name,
            datapath=args.dataset_path,
            batch_size=args.train_batch_size,
            test_batch=1,
            datathread=args.dataloader_num_workers,
            logger=logger)

    # because the optimizer not optimized every time,
    # we need to calculate how many steps it optimizes, it is usually optimized by 
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Learning Rate Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps = args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps = args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with the accelerator
    unet, optimizer, train_loader, test_loader, val_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, test_loader, val_loader, lr_scheduler
    )
    
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        print("Using BF16 Precision.")

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)


    # Here is the DDP training: actually is 4
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps



    logger.info("***** Running training *****")
    logger.info(f"  Tracking with {accelerator.log_with}")
    logger.info(f"  Project Name = {args.tracker_project_name}")
    logger.info(f"  Num examples = {len(train_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        initial_global_step, first_epoch, args.resume_from_checkpoint = resume_from(
           checkpoint=args.resume_from_checkpoint,
           output_dir=args.output_dir,
           accelerator=accelerator,
           update_steps_per_epoch=num_update_steps_per_epoch
        ) 
    else:
        initial_global_step = 0

    logger.info(f"  Learning rate = {lr_scheduler.get_last_lr()[0]}")

    global_step = initial_global_step

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    if accelerator.is_main_process:
        unet.eval()
        # test_path = "/mnt/disks/data1/Masks_Test/"
        # image_paths = [test_path + s for s in os.listdir(test_path)] 
        # run_inference( logger,
        #     vae=vae,
        #     text_encoder=text_encoder,
        #     tokenizer=tokenizer,
        #     unet=unet,
        #     args=args,
        #     accelerator=accelerator,
        #     weight_dtype=weight_dtype,
        #     scheduler=noise_scheduler,
        #     epoch=2,
        #     image_paths=image_paths,
        #     denoise_steps=100,
        #     )
        log_validation( logger,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            args=args,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            scheduler=noise_scheduler,
            epoch=1000,
            denoise_steps=100,
            num_vals=200,
            val_loader = test_loader)
    #     log_validation( logger,
    #         vae=vae,
    #         text_encoder=text_encoder,
    #         tokenizer=tokenizer,
    #         unet=unet,
    #         args=args,
    #         accelerator=accelerator,
    #         weight_dtype=weight_dtype,
    #         scheduler=noise_scheduler,
    #         epoch=-2010,
    #         denoise_steps=10,
    #         num_vals=200,
    #         val_loader = train_loader)


    
    # =================== Training Loop ===================
    for epoch in range(first_epoch, args.num_train_epochs):
        break
        unet.train() 
        train_loss = 0.0
    
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(unet):

                # load image and mask 
                image_data = batch[0]
                mask = batch[1]

                # ==== Resize/shape data for stable diffusion standards ====
                # mask is only a single channel so copy it across 3
                mask_single = mask.unsqueeze(1)
                mask_stacked = mask_single.repeat(1,3,1,1) # dim 0 is batch?
                mask_stacked = mask_stacked.float() # the dataset has it as a float

                image_data_resized = resize_max_res_tensor(image_data,is_disp=False) #range in (0-1)
                mask_resized = resize_max_res_tensor(mask_stacked,is_disp=True) # not range

                # mask normalization: [([1, 3, 432, 768])]
                mask_resized_normalized = Disparity_Normalization(mask_resized)

                if np.random.rand() > .5 :
                    pass

                randhflip = v2.RandomHorizontalFlip(p=0.5)
                flipped_images, flipped_masks = torch.split(randhflip(torch.cat([image_data_resized, mask_resized_normalized],dim=1)),3,dim=1)


                # ==== convert images and masks into latent space. ====
                rgb_latents = encode_to_latent_space(vae, flipped_images, weight_dtype)
                mask_latents = encode_to_latent_space(vae, flipped_masks, weight_dtype)

                # ==== Add noise for diffusion ====
                
                # here is the setting batch size, in our settings, it can be 1.0
                current_batch_size = mask_latents.shape[0]

                # in the Stable Diffusion, the iterations numbers is 1000 for adding the noise and denosing.
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (current_batch_size,), device=mask_latents.device)
                timesteps = timesteps.long()

                # Sample noise that we'll add to the latents
                noise = annealed_pyramid_noise_like(mask_latents, timesteps, total_steps=noise_scheduler.config.num_train_timesteps) # create noise

                
                # add noise to the depth lantents
                noisy_mask_latents = noise_scheduler.add_noise(mask_latents, noise, timesteps)


                # ==== Encode text embedding for empty prompt ====
                empty_text_embed = new_empty_prompt(tokenizer, text_encoder, weight_dtype)
                batch_empty_text_embed = empty_text_embed.repeat((noisy_mask_latents.shape[0], 1, 1))  # [B, 2, 1024]


                # ==== Get the target for loss depending on the prediction type ====
                # TODO: change training noise scheduler to DDPM
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(mask_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                

                # ==== Predict the noise residual and compute the loss. ====
                # This is the concatenation of rgb and mask latents
                # TODO: This is the reverse order of the latents ?
                unet_input = torch.cat([rgb_latents, noisy_mask_latents], dim=1)  # this order is important: [1,8,H,W]
                
                # predict the noise residual
                noise_pred = unet(unet_input, timesteps, 
                                  encoder_hidden_states=batch_empty_text_embed
                                 ).sample  # [B, 4, h, w]
                
                # Loss Function
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                
                # ==== Backpropagate ====
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # currently the EMA is not used.
            if accelerator.sync_gradients:

                if args.use_ema:
                    ema_unet.step(unet.parameters())

                # ==== Update Progress ====
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0
                
                # ==== Saving the checkpoints ====
                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(accelerator, args, logger, global_step)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Stop training
            if global_step >= args.max_train_steps:
                break
        
        

        # ===== Per Epoch =====

        if accelerator.is_main_process:
            # validation each epoch by calculate the epe and the visualization depth
            if args.use_ema:    
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
                
            # validation inference here
            log_validation( 
                logger,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                scheduler=noise_scheduler,
                epoch=epoch,
                val_loader = val_loader,
                denoise_steps=10,
                num_vals=50,
                step=global_step
            )
            
            if args.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())
                

    
        
    # Create the pipeline for training and savet
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
    
    
        
        
    


if __name__=="__main__":
    main()
