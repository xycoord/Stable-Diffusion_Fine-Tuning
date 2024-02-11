from PIL import Image
import torch

import numpy as np
import os
import logging

from Inference.depth_pipeline_half import DepthEstimationPipeline
# from Inference.depth_pipeline import DepthEstimationPipeline


def log_validation(logger, vae,text_encoder,tokenizer,unet,args,accelerator,weight_dtype,scheduler,epoch,
                   input_image_path="/mnt/disks/data1/sceneflow/frames_cleanpass/flythings3d/TEST/A/0000/left/0006.png"
                   ):
    """Run a prediction on a sample input and save the result to see how training is going"""
    denoise_steps = 10
    ensemble_size = 10
    processing_res = 768
    match_input_res = True
    batch_size = 1
    color_map="binary"
    
    
    logger.info("Running validation ... ")
    pipeline = DepthEstimationPipeline.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
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

        pipe_out = pipeline(input_image_pil,
             denosing_steps=denoise_steps,
             ensemble_size= ensemble_size,
             processing_res = processing_res,
             match_input_res = match_input_res,
             batch_size = batch_size,
             color_map = color_map,
             show_progress_bar = True,
             )

        depth_pred: np.ndarray = pipe_out.depth_np
        depth_colored: Image.Image = pipe_out.depth_colored

        # savd as npy
        rgb_name_base = os.path.splitext(os.path.basename(input_image_path))[0]
        pred_name_base = rgb_name_base + "_pred"

        npy_save_path = os.path.join(args.output_dir, f"{pred_name_base}.npy")
        if os.path.exists(npy_save_path):
            logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
        np.save(npy_save_path, depth_pred)

        # Colorize
        colored_save_path = os.path.join(
            args.output_dir, f"{pred_name_base}_{epoch}_colored.png"
        )
        if os.path.exists(colored_save_path):
            logging.warning(
                f"Existing file: '{colored_save_path}' will be overwritten"
            )
        depth_colored.save(colored_save_path)
        
        del depth_colored
        del pipeline
        torch.cuda.empty_cache()
