import argparse

from canvas_config import TRAINABLE_CLASSES, TRAINABLE_THEMES


def get_args():
    parser = argparse.ArgumentParser(
        prog="TrainESD",
        description="Finetuning stable diffusion model to erase concepts using ESD method",
    )
    parser.add_argument(
        "--train_method",
        help="method of training",
        type=str,
        required=True,
        choices=["xattn", "noxattn", "selfattn", "full"],
    )
    parser.add_argument(
        "--start_guidance",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=3,
    )
    parser.add_argument(
        "--negative_guidance",
        help="guidance of negative training used to train",
        type=float,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--iterations",
        help="iterations used to train",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--momentum",
        help="momentum used to train",
        type=float,
        required=False,
        default=0.9,
    )
    parser.add_argument(
        "--weight_decay",
        help="weight decay used to train",
        type=float,
        required=False,
        default=5e-4,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/train_esd.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--data_path",
        help="dataset path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--devices",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="0,0",
    )
    parser.add_argument(
        "--seperator",
        help="separator if you want to train bunch of words separately",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory to save results",
        type=str,
        required=False,
        default="results/style50",
    )

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--w_lr",
        help="learning rate of upper level",
        type=float,
        required=False,
        default=1e-3,
    )
    parser.add_argument(
        "--SignESD",
        help="SignSGD or not",
        action="store_true",
    )
    parser.add_argument(
        "--w_path",
        help="select weight path for SignESD",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--random_choice",
        help="Random Choice or not",
        action="store_true",
    )
    return parser.parse_args()
