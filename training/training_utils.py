import torch

def new_empty_prompt(tokenizer, text_encoder, weight_dtype):
    prompt = ""
    text_inputs = tokenizer(
        prompt,
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(text_encoder.device) #[1,2]
    empty_text_embed = text_encoder(text_input_ids)[0].to(weight_dtype)
    return empty_text_embed


def encode_to_latent_space(vae, input_image, weight_dtype, latent_scale_factor=0.18215):
    h = vae.encoder(input_image.to(weight_dtype))
    moments = vae.quant_conv(h)
    mean, logvar = torch.chunk(moments, 2, dim=1)
    latents = mean * latent_scale_factor
    return latents

