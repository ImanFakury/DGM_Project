from dataclasses import dataclass, field
from typing import Literal, List


@dataclass
class AdaptiveCFGConfig:
    conflict_metric: Literal["l2_norm", "cosine"] = "l2_norm"
    normalization: Literal["sigma", "uncond_norm"] = "sigma"
    mapping_fn: Literal["exponential", "reciprocal"] = "exponential"
    w_min: float = 2.0
    w_max: float = 10.0
    alpha: float = 1.0
    beta: float = 0.5
    epsilon: float = 1e-5

    def to_dict(self):
        return {
            "conflict_metric": self.conflict_metric,
            "normalization": self.normalization,
            "mapping_fn": self.mapping_fn,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
        }


@dataclass
class ExperimentConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    num_inference_steps: int = 50
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    dataset_name: Literal["parti_prompts", "coco_val"] = "coco_val"
    num_samples: int = 50
    output_dir: str = "outputs"
    device: str = "cuda"
    image_size: int = 512
    fixed_cfg_scales: List[float] = field(default_factory=lambda: [3.0, 7.5, 12.0])
