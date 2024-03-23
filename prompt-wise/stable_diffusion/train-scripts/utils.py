import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from stable_diffusion.ldm.util import instantiate_from_config


def load_state_dict(ckpt_path):
    return torch.load(ckpt_path, map_location="cpu")["state_dict"]


def init_model_from_config(config, device="cpu", state_dict=None):
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    if state_dict is not None:
        m, u = model.load_state_dict(state_dict, strict=False)
    return model


def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def save_model(model, output_name):
    torch.save({"state_dict": model.state_dict()}, output_name)


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)
