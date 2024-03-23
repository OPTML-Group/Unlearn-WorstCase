import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lottery Tickets Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="./data", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument(
        "--input_size", type=int, default=32, help="size of input images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".data/tiny-imagenet-200",
        help="dir to tiny-imagenet",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        help="architecture for imagenet size samples",
    )

    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--train_seed",
        default=1,
        type=int,
        help="seed for training (default value same as args.seed)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default="results",
        type=str,
    )
    parser.add_argument("--cp_path", type=str, default=None, help="model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--rewind_epoch", default=0, type=int, help="rewind checkpoint")
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")

    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="w_retrain", help="method to unlearn"
    )
    parser.add_argument(
        "--theta_lr", default=0.01, type=float, help="the learning rate of lower level"
    )
    parser.add_argument(
        "--w_lr", default=0.01, type=float, help="the learning rate of upper level"
    )
    parser.add_argument(
        "--select_epochs",
        default=10,
        type=int,
        help="number of total epochs for select to run",
    )

    parser.add_argument(
        "--unlearn_steps",
        default=10,
        type=int,
        help="number of unroll steps for unlearn to run",
    )

    parser.add_argument(
        "--mode",
        default="optm",
        choices=["optm", "swap", "anlys", "re_optm", "re_swap", "re_anlys", "trans_optm", "trans_swap", "trans_anlys", "trans_kl_anlys"],
        type=str,
        help="selection mode",
    )

    parser.add_argument(
        "--swap_nums",
        default=100,
        type=int,
        help="number of data to be swapped",
    )

    parser.add_argument("--gamma", default=0.0, type=float, help="the ratio of norm")
    parser.add_argument("--alpha", default=1e-3, type=float, help="the ratio of l1-sparse")

    parser.add_argument("--norm", default=2.0, type=float, help="norm of penalty term")
    parser.add_argument("--feq_to_bi", default=20, type=int, help="frequency of coverting w to binary")
    parser.add_argument("--exp", default=False, type=bool, help="use exp in the upper level or not")

    parser.add_argument("--w_path", type=str, default=None, help="select weight")

    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=None,
        help="Number of data to forget",
    )
    parser.add_argument(
        "--class_to_replace", type=int, default=-1, help="Specific class to forget"
    )

    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )

    parser.add_argument("--mask_path", default=None, type=str, help="mask path")
    parser.add_argument('--reverse', action='store_true', help='Reverse the order')
    ##################################### SCRUB setting #################################################

    # https://github.com/ljcc0930/Unlearn-Bench/blob/a173645a2297126ccfbbf453adb31f028ba68945/unlearnbench/unlearn/method/SCRUB.py#L53
    parser.add_argument("--T", default=4, type=float, help="Temperature")
    parser.add_argument("--scrub_gamma", default=0.99, type=float, help="gamma for scrub")
    parser.add_argument("--scrub_alpha", default=0.001, type=float, help="alpha for scrub")
    parser.add_argument("--scrub_beta", default=0.1, type=float, help="beta for scrub")
    parser.add_argument("--m_steps", default=1, type=int, help="m_steps for scrub")
    parser.add_argument("--smoothing", default=0.0, type=float, help="smoothing for scrub")
    parser.add_argument("--lr_decay_rate", default=0.1, type=float, help="lr decay rate")
    parser.add_argument("--lr_decay_epochs", default=[3, 5, 9], type=list, help="lr decay epochs")

    ##################################### Attack setting #################################################
    parser.add_argument(
        "--attack", type=str, default="backdoor", help="method to unlearn"
    )
    parser.add_argument(
        "--trigger_size",
        type=int,
        default=4,
        help="The size of trigger of backdoor attack",
    )
    return parser.parse_args()
