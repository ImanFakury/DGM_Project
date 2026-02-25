import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union
from diffusers import StableDiffusionPipeline, DDIMScheduler
from core.controller import AdaptiveGuidanceController
from config import AdaptiveCFGConfig, ExperimentConfig


def _get_sigma_t(scheduler, t_index):
    if hasattr(scheduler, "sigmas"):
        return float(scheduler.sigmas[t_index].item())
    if hasattr(scheduler, "alphas_cumprod"):
        alphas = scheduler.alphas_cumprod
        idx = min(t_index, len(alphas) - 1)
        alpha_t = alphas[idx].item()
        sigma_t = ((1 - alpha_t) / alpha_t) ** 0.5
        return float(sigma_t)
    return 1.0


class AdaptiveSamplerWrapper:
    def __init__(
        self,
        pipeline: StableDiffusionPipeline,
        adaptive_config: Optional[AdaptiveCFGConfig] = None,
        fixed_guidance_scale: Optional[float] = None,
    ):
        self.pipeline = pipeline
        self.adaptive_config = adaptive_config
        self.fixed_guidance_scale = fixed_guidance_scale
        self.is_adaptive = adaptive_config is not None
        self.device = pipeline.device

    def _encode_prompt(self, prompt: str, negative_prompt: str = ""):
        tokenizer = self.pipeline.tokenizer
        text_encoder = self.pipeline.text_encoder

        def _encode(text):
            tokens = tokenizer(
                text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                emb = text_encoder(tokens.input_ids.to(self.device))[0]
            return emb

        cond_emb = _encode(prompt)
        uncond_emb = _encode(negative_prompt)
        return cond_emb, uncond_emb

    @torch.no_grad()
    def sample(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        seed: int = 42,
        height: int = 512,
        width: int = 512,
        negative_prompt: str = "",
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        scheduler = self.pipeline.scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        cond_emb, uncond_emb = self._encode_prompt(prompt, negative_prompt)

        latents = torch.randn(
            (1, self.pipeline.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=self.pipeline.unet.dtype,
        )
        latents = latents * scheduler.init_noise_sigma

        controller = None
        if self.is_adaptive:
            controller = AdaptiveGuidanceController(self.adaptive_config)

        w_history = []

        for i, t in enumerate(timesteps):
            latent_input = scheduler.scale_model_input(latents, t)

            with torch.no_grad():
                noise_uncond = self.pipeline.unet(
                    latent_input, t, encoder_hidden_states=uncond_emb
                ).sample
                noise_cond = self.pipeline.unet(
                    latent_input, t, encoder_hidden_states=cond_emb
                ).sample

            if self.is_adaptive and controller is not None:
                sigma_t = _get_sigma_t(scheduler, i)
                w_t, s_t = controller.update_and_get_w(noise_cond, noise_uncond, sigma_t)
                w_history.append(w_t)
            else:
                w_t = self.fixed_guidance_scale
                w_history.append(w_t)

            noise_pred = noise_uncond + w_t * (noise_cond - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / self.pipeline.vae.config.scaling_factor
        with torch.no_grad():
            image_tensor = self.pipeline.vae.decode(latents).sample

        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

        return image_pil, w_history


def build_pipeline(exp_config: ExperimentConfig) -> StableDiffusionPipeline:
    scheduler = DDIMScheduler.from_pretrained(
        exp_config.model_id, subfolder="scheduler"
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        exp_config.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if exp_config.device == "cuda" else torch.float32,
    )
    pipeline = pipeline.to(exp_config.device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=True)
    return pipeline
