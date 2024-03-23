import copy
import os
from collections import OrderedDict
import matplotlib.pyplot as plt

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils

import numpy as np
from trainer import validate
from torch.utils.data import DataLoader


def norm_grad(x, p):
    norm_value = torch.norm(x, p=p)
    return (x.abs()**(p - 1)) * x.sign() / norm_value**(p - 1)


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def main():
    args = arg_parser.parse_args()

    if args.feq_to_bi > args.select_epochs:
        args.feq_to_bi = args.select_epochs

    gaps = {"UA": []}
    w_records = []
    bi_w_records = []
    w_norms = []
    bi_w_norms = []
    select_epoch_losses = []

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
    ) = utils.setup_model_indexdataset(args)
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

    mask = None
    if args.mask_path:
        mask = torch.load(args.mask_path)

    criterion = nn.CrossEntropyLoss(reduction="none")
    train_full_loader, forget_loader, remain_loader, w = utils.update_w(train_set,
                                                    class_to_replace=args.class_to_replace, 
                                                    num_indexes_to_replace=args.num_indexes_to_replace,
                                                    seed=args.seed,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    args=args
                                                    )
    w = w.cuda()
    evaluation_result = None

    for epoch in range(args.select_epochs):

        if "retrain" in args.unlearn:
            print("->->->->->->->->->-> Randomly initialize model <-<-<-<-<-<-<-<-<-<-")
            model.apply(utils.weights_init)
        else:
            print("->->->->->->->->->-> Reload model <-<-<-<-<-<-<-<-<-<-")
            model.load_state_dict(checkpoint, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        if args.unlearn == "w_RL" or args.unlearn == "w_boundary_shrink" or args.unlearn == "w_boundary_expanding" or args.unlearn == "w_scrub":
            pre_data_loaders = OrderedDict(
                retain=remain_loader, forget=forget_loader, val=val_loader, test=test_loader
            )
            unlearn_method(pre_data_loaders, model, criterion, args, w, mask)

        else:
            unlearn_method(train_full_loader, model, criterion, args, w, mask)

        if args.mode == "optm":
            w, select_epoch_loss = unlearn.optimize_select(train_full_loader, model, criterion, args, w)
        elif args.mode == "re_optm":
            w, select_epoch_loss = unlearn.reverse_optimize_select(train_full_loader, model, criterion, args, w)

        select_epoch_losses.append(select_epoch_loss.item())

        topk_indices = torch.topk(w, args.num_indexes_to_replace)[1]
        bi_w = torch.zeros_like(w)
        bi_w[topk_indices] = 1

        w_records.append(w.cpu().numpy())
        bi_w_records.append(bi_w.cpu().numpy())

        _, forget_loader, remain_loader, _ = utils.update_w(train_set,
                                    w=bi_w,
                                    class_to_replace=args.class_to_replace, 
                                    seed=args.seed,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    args=args)

        if (epoch + 1) % args.feq_to_bi == 0:
            w = bi_w
            if epoch == args.select_epochs - 1:
                _, forget_loader, remain_loader, _ = utils.update_w(train_set,
                                                                    w=w,
                                                                    class_to_replace=args.class_to_replace, 
                                                                    seed=args.seed,
                                                                    batch_size=args.batch_size,
                                                                    shuffle=True,
                                                                    args=args)

    w_path = os.path.join(args.save_dir, "select_weight.pth.tar")
    gaps['w'] = w_records
    gaps['bi_w'] = bi_w_records
    gaps['loss'] = select_epoch_losses
    torch.save(gaps, w_path)

    # Evaluate 
    unlearn_data_loaders = OrderedDict(
        retain=remain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    forget_dataset = forget_loader.dataset
    retain_dataset = remain_loader.dataset

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, nn.CrossEntropyLoss(), args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

if __name__ == "__main__":
    main()