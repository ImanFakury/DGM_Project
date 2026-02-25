import json
import os
import random
from typing import List
from datasets import load_dataset


PARTI_FAILURE_SUBSET = [
    "a close-up portrait of a person smiling",
    "two hands holding a coffee cup",
    "a sign that reads OPEN in red letters",
    "a brick wall with intricate texture pattern",
    "a group of five people standing together",
    "a face with detailed eyes and freckles",
    "the word HELLO written in graffiti",
    "a highly detailed human hand with five fingers",
    "a mosaic pattern with small repeating tiles",
    "a face in profile with sharp lighting",
]


def load_prompts(
    dataset_name: str = "coco_val",
    num_samples: int = 50,
    seed: int = 0,
) -> List[str]:
    random.seed(seed)

    if dataset_name == "parti_prompts":
        return _load_parti_prompts(num_samples)
    elif dataset_name == "coco_val":
        return _load_coco_prompts(num_samples)
    elif dataset_name == "failure_subset":
        return PARTI_FAILURE_SUBSET[:num_samples]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_coco_prompts(num_samples: int) -> List[str]:
    try:
        dataset = load_dataset("HuggingFaceM4/COCO", split="validation", streaming=True)
        prompts = []
        for item in dataset:
            if len(prompts) >= num_samples:
                break
            captions = item.get("captions", {})
            if isinstance(captions, dict):
                text_list = captions.get("text", [])
                if text_list:
                    prompts.append(text_list[0])
            elif isinstance(captions, list) and captions:
                prompts.append(captions[0])
        if not prompts:
            raise RuntimeError("Empty COCO load")
        return prompts[:num_samples]
    except Exception:
        fallback = [
            "a dog running in a park",
            "a cat sitting on a windowsill",
            "a red car parked on a street",
            "a bowl of fruit on a wooden table",
            "a person riding a bicycle",
            "a sunset over the ocean",
            "a mountain covered in snow",
            "a city skyline at night",
            "children playing in a playground",
            "a plate of pasta with tomato sauce",
        ]
        extended = (fallback * ((num_samples // len(fallback)) + 1))[:num_samples]
        return extended


def _load_parti_prompts(num_samples: int) -> List[str]:
    try:
        dataset = load_dataset("nateraw/parti-prompts", split="train")
        prompts = [item["Prompt"] for item in dataset]
        random.shuffle(prompts)
        return prompts[:num_samples]
    except Exception:
        return PARTI_FAILURE_SUBSET[:num_samples]
