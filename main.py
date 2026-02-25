import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AdaptiveCFGConfig, ExperimentConfig
from dataset import load_prompts
from core.sampler_bridge import build_pipeline
from experiments.manager import AblationStudy
from analyze import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive CFG Experiment Runner")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--dataset", type=str, default="coco_val",
                        choices=["coco_val", "parti_prompts", "failure_subset"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--w_min", type=float, default=2.0)
    parser.add_argument("--w_max", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_ablations", action="store_true")
    parser.add_argument("--only_analyze", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.only_analyze:
        generate_all_plots(args.output_dir)
        return

    exp_config = ExperimentConfig(
        model_id=args.model_id,
        num_inference_steps=args.num_steps,
        seeds=args.seeds,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device,
        fixed_cfg_scales=[3.0, 7.5, 12.0],
    )

    base_cfg_config = AdaptiveCFGConfig(
        w_min=args.w_min,
        w_max=args.w_max,
        alpha=args.alpha,
        beta=args.beta,
    )

    print(f"Loading prompts from: {args.dataset} ({args.num_samples} samples)")
    prompts = load_prompts(args.dataset, args.num_samples)
    print(f"Loaded {len(prompts)} prompts")

    print(f"Loading pipeline: {args.model_id}")
    pipeline = build_pipeline(exp_config)

    study = AblationStudy(exp_config, base_cfg_config)

    if not args.skip_baselines:
        print("=== Running Fixed CFG Baselines ===")
        study.run_baselines(prompts, pipeline)

    if not args.skip_ablations:
        print("=== Running Adaptive CFG Ablations ===")
        study.run_adaptive_variants(prompts, pipeline)

    results = study.save_results()

    print("\n=== Results Summary ===")
    for variant, metrics in results.items():
        clip = metrics.get("clip_score", 0)
        hf = metrics.get("hf_energy_mean", 0)
        div = metrics.get("lpips_diversity", 0)
        print(f"  {variant:40s} | CLIP={clip:.2f} | HF={hf:.1f} | LPIPS={div:.3f}")

    print("\n=== Generating Plots ===")
    generate_all_plots(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
