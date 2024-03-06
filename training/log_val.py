from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from utils.image_util import resize_max_res,chw2hwc,colorize_depth_maps

import numpy as np
import os
import logging
import matplotlib.pyplot as plt

from Inference.depth_pipeline_half import DepthEstimationPipeline
# from Inference.depth_pipeline import DepthEstimationPipeline


def log_validation(logger, 
                   vae,text_encoder,tokenizer,unet,
                   args,
                   accelerator,weight_dtype,scheduler,
                   epoch,
                   val_loader,
                   input_image_path="/mnt/disks/data1/sceneflow/frames_cleanpass/flythings3d/TEST/A/0000/left/0006.png",
                   step=0,
                   denoise_steps = 10,
                   num_vals = 20
                   ):
    """Run a prediction on a sample input and save the result to see how training is going"""
    ensemble_size = 10
    processing_res = 768
    match_input_res = True
    batch_size = 1
    color_map="binary"
    
    logger.info("Running validation ... ")
    
    pipeline = DepthEstimationPipeline.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet = accelerator.unwrap_model(unet),
        safety_checker=None,
        scheduler = accelerator.unwrap_model(scheduler),
        )

    pipeline = pipeline.to(accelerator.device)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass  

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        input_image_pil = Image.open(input_image_path)
        # target_image_pil = Image.open("/mnt/disks/data1/Trans10K/validation/easy/masks/67_mask.png")
        # target = pil_to_tensor(target_image_pil)[0].float()/255


        loss_total = 0

        for val_index, batch in enumerate(val_loader):
            if val_index >= num_vals:
                break
            inputs, targets, filenames = batch


            pipe_out = pipeline(
                input_image_pil,
                input_image_tensor = inputs[0],
                ensemble_size= ensemble_size,
                processing_res = processing_res,
                match_input_res = match_input_res,
                batch_size = batch_size,
                color_map = color_map,
                show_progress_bar = True,
                denoise_steps = denoise_steps
            )

            depth_pred: np.ndarray = pipe_out.depth_np
            depth_colored: Image.Image = pipe_out.depth_colored

            target = targets[0].cpu().numpy().astype(np.float32)
            target = target.clip(0, 1)
            target_colored = colorize_depth_maps(
                target, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            target_colored = (target_colored * 255).astype(np.uint8)
            target_colored_hwc = chw2hwc(target_colored)
            target_img = Image.fromarray(target_colored_hwc)

            input = inputs[0].cpu().numpy().astype(np.float32)
            input = input.clip(0, 1)
            input = (input * 255).astype(np.uint8)
            input_hwc = chw2hwc(input)
            input_img = Image.fromarray(input_hwc)


            loss = F.mse_loss(torch.from_numpy(depth_pred).to(targets.device), targets[0], reduction="mean")
            loss_total += loss

            epoch_dir = os.path.join(args.output_dir, f"{epoch}/")
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)

            # savd as npy
            rgb_name_base = os.path.splitext(os.path.basename(filenames[0]))[0]
            pred_name_base = rgb_name_base + "_pred_"

            npy_save_path = os.path.join(epoch_dir, f"{pred_name_base}_{epoch}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, depth_pred)

            # Colorize
            colored_save_path = os.path.join(epoch_dir, f"{pred_name_base}_{epoch}.png")
            if os.path.exists(colored_save_path):
                logging.warning(
                    f"Existing file: '{colored_save_path}' will be overwritten"
                )
            depth_colored.save(colored_save_path)
            
            target_save_path = os.path.join(epoch_dir, f"{rgb_name_base}_target.png")
            if os.path.exists(target_save_path):
                logging.warning(
                    f"Existing file: '{target_save_path}' will be overwritten"
                )
            target_img.save(target_save_path)

            input_save_path = os.path.join(epoch_dir, f"{rgb_name_base}_input.png")
            if os.path.exists(input_save_path):
                logging.warning(
                    f"Existing file: '{input_save_path}' will be overwritten"
                )
            input_img.save(input_save_path)


        accelerator.log({"valid_loss": loss_total/num_vals}, step=step)

        del depth_colored
        del pipeline
        torch.cuda.empty_cache()
