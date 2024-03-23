import os
import sys

import torch

import wandb

sys.path.append(".")
import arg_parser
import dataset
import esd
from canvas_config import TRAINABLE_CLASSES, TRAINABLE_THEMES

import selection
import itertools


def get_prompts():   
    themes = ["Crayon", "Ukiyoe", "Mosaic", "Sketch", "Dadaism", "Winter", "Van_Gogh", "Rust", "Glowing_Sunset", "Red_Blue_Ink"]
    objects = ["Statues", "Towers", "Human", "Flowers", "Birds", "Trees", "Waterfalls", "Jellyfish", "Sandwiches", "Dogs"]
    prompts = ['A painting of a {} in {} Style.'.format(obj, theme) for obj, theme in itertools.product(objects, themes)]
    return prompts


if __name__ == "__main__":
    args = arg_parser.get_args()

    if args.dry_run:
        wandb = None
    else:
        wandb.init(
            project="object in style",
            name=args.object,
            config=args,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_dir
    print(f"Saving the model to {output_name}")

    prompts = get_prompts()
    print(f"Prompt for unlearning: {prompts}")

    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    devices = [f"cuda:{int(d.strip())}" for d in args.devices.split(",")]
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    if args.SignESD:
        unlearn_method = esd.SignESD(
            train_method=train_method,
            start_guidance=start_guidance,
            negative_guidance=negative_guidance,
            iterations=iterations,
            eval_iters=10,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            output_name=output_name,
            image_size=image_size,
            ddim_steps=ddim_steps,
            wandb=wandb,
        )
    else:
        unlearn_method = esd.ESD(
            train_method=train_method,
            start_guidance=start_guidance,
            negative_guidance=negative_guidance,
            iterations=iterations,
            eval_iters=10,
            lr=lr,
            output_name=output_name,
            image_size=image_size,
            ddim_steps=ddim_steps,
            wandb=wandb,
        )

    eval_set = dataset.GenerationDataset(
        path=args.data_path,
        split="train",
        splits=(0.9, 0.05, 0.05),
        res=256,
        crop_res=256,
        flip_prob=0.0,
        themes=["Crayon", "Ukiyoe", "Mosaic", "Sketch", "Dadaism", "Winter", "Van_Gogh", "Rust", "Glowing_Sunset", "Red_Blue_Ink"],
        classes=["Statues", "Towers", "Human", "Flowers", "Birds", "Trees", "Waterfalls", "Jellyfish", "Sandwiches", "Dogs"],
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=4, shuffle=False, num_workers=20
    )

    eval_method = esd.Eval(eval_loader=eval_loader)
    env = selection.DataSelection(
        prompts,
        config_path,
        ckpt_path,
        devices,
        unlearn_method,
        eval_method,
        w_lr=args.w_lr,
        gamma=1e-4,
        wandb=wandb,
        save_path=args.output_dir
    )
    w = env.optimize()