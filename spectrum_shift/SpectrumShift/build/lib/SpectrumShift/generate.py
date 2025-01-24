import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffusers.utils.torch_utils import randn_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_views(noisy_images, views):
  noisy_image_views = [view.view(noisy_images[0]) for view in views]
  return torch.stack(noisy_image_views)

def apply_inv_views(noise, views):
  inverted_preds = [view.inverse_view(pred) for pred, view in zip(noise, views)]
  return torch.stack(inverted_preds) 

@torch.no_grad()
def generate_stage_1(model, prompt_embeds, neg_prompt_embeds, views, num_inference_steps=200, guidance_scale=6, reduction='mean', generator=None):
  # Set parameters
  num_images_per_prompt = 1 
  height = model.unet.config.sample_size
  width = model.unet.config.sample_size
  num_prompts = prompt_embeds.shape[0]

  model.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
  timesteps = model.scheduler.timesteps

  # Cat prompt embeds for CFG 
  prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])

  noisy_images = model.prepare_intermediate_images(
    num_images_per_prompt,
    model.unet.config.in_channels,
    height,
    width,
    prompt_embeds.dtype,
    DEVICE,
    generator
  )

  for timestep in tqdm(timesteps):
    noisy_image_views = apply_views(noisy_images, views)

    # Generate model inputs for both neg(uncond) and positive prompts
    model_input = torch.cat([noisy_image_views] * 2)
    model_input = model.scheduler.scale_model_input(model_input, timestep)

    # Predict noise estimates from model
    noise_pred = model.unet(
      model_input,
      timestep,
      encoder_hidden_states = prompt_embeds,
      cross_attention_kwargs = None,
      return_dict = False
    )[0]
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

    # Generate inverse view of noise from both uncond and text prompts
    noise_pred_uncond = apply_inv_views(noise_pred_uncond, views)
    noise_pred_text = apply_inv_views(noise_pred_text, views)
    
    # Split into noise estimate and variance estimates
    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Reduce predicted noise and variances via mean
    noise_pred = noise_pred.view(-1, num_prompts, 3, 64, 64)
    predicted_variance = predicted_variance.view(-1, num_prompts, 3, 64, 64)
    
    noise_pred = noise_pred.mean(1)
    predicted_variance = predicted_variance.mean(1)

    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

    # Compute x_{t-1} from x_t
    noisy_images = model.scheduler.step(
        noise_pred, 
        timestep, 
        noisy_images, 
        generator = generator, 
        return_dict = False
    )[0]

  return noisy_images