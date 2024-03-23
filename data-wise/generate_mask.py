import copy
import os
from collections import OrderedDict

import arg_parser
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils

from torch.utils.data import DataLoader

def save_gradient_ratio(data_loaders, model, criterion, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.theta_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}

    forget_loader = data_loaders["forget"]
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, (image, target) in enumerate(forget_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = - criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]
            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        torch.save(hard_dict, os.path.join(args.save_dir, "with_{}.pt".format(i)))


def main():
    args = arg_parser.parse_args()

    if args.feq_to_bi > args.select_epochs:
        args.feq_to_bi = args.select_epochs

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)

    # prepare dataset
    (
        model,
        train_set, 
        valid_set, 
        test_set
    ) = utils.setup_model_dataset(args)
    model.cuda()

    val_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False
    )

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.cp_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if "retrain" not in args.unlearn:
            model.load_state_dict(checkpoint, strict=True)

    criterion = nn.CrossEntropyLoss(reduction="none")
    _, forget_loader, remain_loader, w = utils.update_w(train_set,
                                                    class_to_replace=args.class_to_replace, 
                                                    num_indexes_to_replace=args.num_indexes_to_replace,
                                                    seed=args.seed,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    args=args
                                                    )
    w = w.cuda()

    unlearn_data_loaders = OrderedDict(
        retain=remain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    save_gradient_ratio(unlearn_data_loaders, model, criterion, args)


if __name__ == "__main__":
    main()
