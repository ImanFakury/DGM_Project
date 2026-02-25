import torch
import torch.nn.functional as F
from config import AdaptiveCFGConfig


class AdaptiveGuidanceController:
    def __init__(self, config: AdaptiveCFGConfig):
        self.config = config
        self.prev_w = config.w_max
        self._metric_fn = self._build_metric_fn()
        self._mapping_fn = self._build_mapping_fn()

    def reset(self):
        self.prev_w = self.config.w_max

    def _build_metric_fn(self):
        cfg = self.config

        def l2_metric(noise_cond, noise_uncond, sigma_t):
            delta = noise_cond - noise_uncond
            delta_norm = torch.norm(delta.reshape(delta.shape[0], -1), dim=-1).mean()
            if cfg.normalization == "sigma":
                denom = sigma_t + cfg.epsilon
            else:
                uncond_norm = torch.norm(
                    noise_uncond.reshape(noise_uncond.shape[0], -1), dim=-1
                ).mean()
                denom = uncond_norm + cfg.epsilon
            return (delta_norm / denom).item()

        def cosine_metric(noise_cond, noise_uncond, sigma_t):
            c_flat = noise_cond.reshape(noise_cond.shape[0], -1)
            u_flat = noise_uncond.reshape(noise_uncond.shape[0], -1)
            cos_sim = F.cosine_similarity(c_flat, u_flat, dim=-1).mean()
            s_t = 1.0 - cos_sim.item()
            if cfg.normalization == "sigma":
                denom = sigma_t + cfg.epsilon
                delta = noise_cond - noise_uncond
                raw = torch.norm(delta.reshape(delta.shape[0], -1), dim=-1).mean().item()
                s_t = s_t * raw / denom
            return s_t

        if self.config.conflict_metric == "l2_norm":
            return l2_metric
        return cosine_metric

    def _build_mapping_fn(self):
        cfg = self.config

        def exp_map(s_t):
            raw = cfg.w_min + (cfg.w_max - cfg.w_min) * torch.exp(
                torch.tensor(-cfg.alpha * s_t)
            ).item()
            return float(torch.clamp(torch.tensor(raw), cfg.w_min, cfg.w_max).item())

        def reciprocal_map(s_t):
            raw = cfg.w_min + (cfg.w_max - cfg.w_min) / (1.0 + cfg.alpha * s_t)
            return float(torch.clamp(torch.tensor(raw), cfg.w_min, cfg.w_max).item())

        if self.config.mapping_fn == "exponential":
            return exp_map
        return reciprocal_map

    def update_and_get_w(self, noise_cond, noise_uncond, sigma_t):
        s_t = self._metric_fn(noise_cond, noise_uncond, sigma_t)
        w_t = self._mapping_fn(s_t)
        w_smoothed = self.config.beta * self.prev_w + (1.0 - self.config.beta) * w_t
        w_smoothed = float(
            torch.clamp(
                torch.tensor(w_smoothed), self.config.w_min, self.config.w_max
            ).item()
        )
        self.prev_w = w_smoothed
        return w_smoothed, s_t
