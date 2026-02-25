import os
import json
import csv
from typing import List, Dict, Any
from PIL import Image

from config import AdaptiveCFGConfig, ExperimentConfig
from core.sampler_bridge import AdaptiveSamplerWrapper, build_pipeline
from eval.metrics import QualityEvaluator


ABLATION_VARIANTS = [
    {
        "name": "adaptive_l2_sigma_exp",
        "conflict_metric": "l2_norm",
        "normalization": "sigma",
        "mapping_fn": "exponential",
    },
    {
        "name": "adaptive_cosine_sigma_exp",
        "conflict_metric": "cosine",
        "normalization": "sigma",
        "mapping_fn": "exponential",
    },
    {
        "name": "adaptive_l2_uncond_exp",
        "conflict_metric": "l2_norm",
        "normalization": "uncond_norm",
        "mapping_fn": "exponential",
    },
    {
        "name": "adaptive_l2_sigma_recip",
        "conflict_metric": "l2_norm",
        "normalization": "sigma",
        "mapping_fn": "reciprocal",
    },
]


class AblationStudy:
    def __init__(self, exp_config: ExperimentConfig, base_cfg_config: AdaptiveCFGConfig):
        self.exp_config = exp_config
        self.base_cfg_config = base_cfg_config
        self.evaluator = QualityEvaluator(device=exp_config.device)
        self.results: Dict[str, Dict[str, Any]] = {}

        os.makedirs(exp_config.output_dir, exist_ok=True)

    def _run_single_variant(
        self,
        pipeline,
        name: str,
        prompts: List[str],
        adaptive_config: AdaptiveCFGConfig = None,
        fixed_scale: float = None,
    ) -> Dict[str, Any]:
        wrapper = AdaptiveSamplerWrapper(
            pipeline=pipeline,
            adaptive_config=adaptive_config,
            fixed_guidance_scale=fixed_scale,
        )

        images_per_seed: List[List[Image.Image]] = []
        for seed in self.exp_config.seeds:
            seed_images = []
            for prompt in prompts:
                img, w_hist = wrapper.sample(
                    prompt=prompt,
                    num_inference_steps=self.exp_config.num_inference_steps,
                    seed=seed,
                    height=self.exp_config.image_size,
                    width=self.exp_config.image_size,
                )
                seed_images.append(img)
                self._save_image(img, name, seed, prompt)
            images_per_seed.append(seed_images)

        metrics = self.evaluator.evaluate_run(images_per_seed, prompts)
        metrics["variant"] = name
        return metrics

    def _save_image(self, img: Image.Image, variant: str, seed: int, prompt: str):
        variant_dir = os.path.join(self.exp_config.output_dir, variant, f"seed_{seed}")
        os.makedirs(variant_dir, exist_ok=True)
        safe_prompt = prompt[:40].replace(" ", "_").replace("/", "_")
        img.save(os.path.join(variant_dir, f"{safe_prompt}.png"))

    def run_baselines(self, prompts: List[str], pipeline):
        for scale in self.exp_config.fixed_cfg_scales:
            name = f"fixed_cfg_{scale}"
            print(f"Running baseline: {name}")
            result = self._run_single_variant(
                pipeline=pipeline,
                name=name,
                prompts=prompts,
                fixed_scale=scale,
            )
            self.results[name] = result

    def run_adaptive_variants(self, prompts: List[str], pipeline):
        for variant_def in ABLATION_VARIANTS:
            name = variant_def["name"]
            print(f"Running ablation: {name}")
            cfg = AdaptiveCFGConfig(
                conflict_metric=variant_def["conflict_metric"],
                normalization=variant_def["normalization"],
                mapping_fn=variant_def["mapping_fn"],
                w_min=self.base_cfg_config.w_min,
                w_max=self.base_cfg_config.w_max,
                alpha=self.base_cfg_config.alpha,
                beta=self.base_cfg_config.beta,
                epsilon=self.base_cfg_config.epsilon,
            )
            result = self._run_single_variant(
                pipeline=pipeline,
                name=name,
                prompts=prompts,
                adaptive_config=cfg,
            )
            self.results[name] = result

    def save_results(self):
        json_path = os.path.join(self.exp_config.output_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)

        csv_path = os.path.join(self.exp_config.output_dir, "results.csv")
        if self.results:
            keys = list(next(iter(self.results.values())).keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in self.results.values():
                    writer.writerow(row)

        print(f"Results saved to {self.exp_config.output_dir}")
        return self.results
