from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from canvas_config import ALL_CLASSES, ALL_THEMES
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CenterSquareCrop:
    def __call__(self, img):
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        right = (w + min_dim) / 2
        bottom = (h + min_dim) / 2
        return F.crop(img, top=int(top), left=int(left), height=min_dim, width=min_dim)


class GenerationDataset(Dataset):
    def __init__(
        self,
        path: str = "../data/quick-canvas-benchmark",
        split: str = "train",
        splits: Tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,  # 256
        crop_res: int = 256,  # 256
        flip_prob: float = 0.0,  # 0.5 for train, 0.0 for val and test
        themes: Tuple[str] = ALL_THEMES,
        classes: Tuple[str] = ALL_CLASSES,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        self.themes = themes
        self.classes = classes
        self.theme_total = len(themes)
        self.class_total = len(classes)
        self.image_per_class = int(20 * splits[0])
        self.trainable_images = (
            self.theme_total * self.class_total * self.image_per_class
        )
        print(f"Total trainable images: {self.trainable_images}")

        self.transforms = transforms.Compose(
            [
                CenterSquareCrop(),
                transforms.Resize((res, res)),
                transforms.RandomCrop((crop_res, crop_res)),
                transforms.RandomHorizontalFlip(p=flip_prob),
            ]
        )

    def __len__(self) -> int:
        return self.trainable_images

    def __getitem__(self, i: int) -> Dict[str, Any]:
        comb_idx = i // self.image_per_class
        theme_idx = comb_idx // self.class_total
        class_idx = comb_idx % self.class_total
        image_idx = i % self.image_per_class

        theme_name = self.themes[theme_idx]
        class_name = self.classes[class_idx]

        image_dir = Path(self.path).joinpath(theme_name)

        # The randomly selected images
        image_1 = Image.open(image_dir.joinpath(f"{class_name}/{image_idx + 1}.jpg"))
        image_1 = self.transforms(image_1)

        # Convert the images to tensors
        image_1 = rearrange(
            2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w"
        )
        if theme_name == "Seed_Images":
            prompt = f"A {class_name} image in Photo style."
        else:
            prompt = f"A {class_name} image in {theme_name.replace('_', ' ')} style."

        # return dict(edited=image_1, edit=dict(c_crossattn=prompt))
        return [image_1, prompt, comb_idx]
