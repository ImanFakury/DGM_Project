import torch
import numpy as np
from PIL import Image
from typing import List, Dict
import torchvision.transforms as T
import torch.nn.functional as F


def _pil_to_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])
    return transform(img)


class CLIPScorer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
                self.device
            )
            self._model.eval()
            self._processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

    def score(self, images: List[Image.Image], prompts: List[str]) -> List[float]:
        self._load()
        scores = []
        for img, prompt in zip(images, prompts):
            inputs = self._processor(
                text=[prompt], images=[img], return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            score = outputs.logits_per_image.squeeze().item()
            scores.append(score)
        return scores

    def mean_score(self, images: List[Image.Image], prompts: List[str]) -> float:
        return float(np.mean(self.score(images, prompts)))


class LPIPSDiversity:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._lpips_fn = None

    def _load(self):
        if self._lpips_fn is None:
            import lpips
            self._lpips_fn = lpips.LPIPS(net="alex").to(self.device)
            self._lpips_fn.eval()

    def _to_lpips_tensor(self, img: Image.Image) -> torch.Tensor:
        t = T.Compose([T.Resize((256, 256)), T.ToTensor()])(img)
        t = t * 2.0 - 1.0
        return t.unsqueeze(0).to(self.device)

    def pairwise_diversity(self, image_sets: List[List[Image.Image]]) -> float:
        self._load()
        distances = []
        for imgs in image_sets:
            if len(imgs) < 2:
                continue
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    t1 = self._to_lpips_tensor(imgs[i])
                    t2 = self._to_lpips_tensor(imgs[j])
                    with torch.no_grad():
                        d = self._lpips_fn(t1, t2).item()
                    distances.append(d)
        return float(np.mean(distances)) if distances else 0.0


class ArtifactEvaluator:
    def saturation_stats(self, images: List[Image.Image]) -> Dict[str, float]:
        sat_values = []
        for img in images:
            img_hsv = img.convert("HSV")
            arr = np.array(img_hsv).astype(float)
            s_channel = arr[:, :, 1]
            sat_values.append(s_channel.std())
        return {
            "saturation_std_mean": float(np.mean(sat_values)),
            "saturation_std_max": float(np.max(sat_values)),
        }

    def high_frequency_energy(self, images: List[Image.Image]) -> Dict[str, float]:
        hf_energies = []
        for img in images:
            gray = np.array(img.convert("L")).astype(float)
            fft = np.fft.fft2(gray)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            r = min(h, w) // 4
            y_idx, x_idx = np.ogrid[:h, :w]
            mask = (y_idx - cy) ** 2 + (x_idx - cx) ** 2 > r ** 2
            hf_energy = float(magnitude[mask].mean())
            hf_energies.append(hf_energy)
        return {
            "hf_energy_mean": float(np.mean(hf_energies)),
            "hf_energy_std": float(np.std(hf_energies)),
        }

    def evaluate(self, images: List[Image.Image]) -> Dict[str, float]:
        sat = self.saturation_stats(images)
        hf = self.high_frequency_energy(images)
        return {**sat, **hf}


class QualityEvaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clip_scorer = CLIPScorer(device)
        self.lpips_diversity = LPIPSDiversity(device)
        self.artifact_eval = ArtifactEvaluator()

    def evaluate_run(
        self,
        images_per_seed: List[List[Image.Image]],
        prompts: List[str],
    ) -> Dict[str, float]:
        all_images = [img for seed_imgs in images_per_seed for img in seed_imgs]

        clip_score = self.clip_scorer.mean_score(all_images, prompts * len(images_per_seed))
        diversity = self.lpips_diversity.pairwise_diversity(
            [[seed_imgs[i] for seed_imgs in images_per_seed] for i in range(len(prompts))]
        )
        artifacts = self.artifact_eval.evaluate(all_images)

        return {
            "clip_score": clip_score,
            "lpips_diversity": diversity,
            **artifacts,
        }
