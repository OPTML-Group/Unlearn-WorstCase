import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils

from trainer import validate
from torch.utils.data import DataLoader


def main():
    args = arg_parser.parse_args()
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

    if args.w_path:
        print("->->->->->->->->->-> Successfully load w! <-<-<-<-<-<-<-<-<-<-")

        w = torch.load(args.w_path)["w"][-1]
        w = torch.from_numpy(w)
        
        if args.reverse:
            w = -w
        
        topk_indices = torch.topk(w, args.num_indexes_to_replace)[1]
        new_w = torch.zeros_like(w)
        new_w[topk_indices] = 1
        w = new_w

    else:
        w = None

    _, forget_loader, remain_loader, w = utils.update_w(train_set,
                                                    w=w,
                                                    class_to_replace=args.class_to_replace, 
                                                    num_indexes_to_replace=args.num_indexes_to_replace,
                                                    seed=args.seed,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    args=args
                                                    )

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

    unlearn_data_loaders = OrderedDict(
        retain=remain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    forget_dataset = forget_loader.dataset
    retain_dataset = remain_loader.dataset

    print("The number of Retain Dataset is {}".format(len(forget_loader.dataset)))
    print("The number of Forget Dataset is {}".format(len(remain_loader.dataset)))

    mask = None
    if args.mask_path:
        mask = torch.load(args.mask_path)
        
    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

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

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args, mask)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
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

    # """training privacy MIA:
    #     in distribution: retain
    #     out of distribution: test
    #     target: (retain, test)"""
    # if "SVC_MIA_training_privacy" not in evaluation_result:
    #     test_len = len(test_loader.dataset)
    #     retain_len = len(retain_dataset)
    #     num = test_len // 2

    #     utils.dataset_convert_to_test(retain_dataset, args)
    #     utils.dataset_convert_to_test(forget_loader, args)
    #     utils.dataset_convert_to_test(test_loader, args)

    #     shadow_train = torch.utils.data.Subset(retain_dataset, list(range(num)))
    #     target_train = torch.utils.data.Subset(
    #         retain_dataset, list(range(num, retain_len))
    #     )
    #     shadow_test = torch.utils.data.Subset(test_loader.dataset, list(range(num)))
    #     target_test = torch.utils.data.Subset(
    #         test_loader.dataset, list(range(num, test_len))
    #     )

    #     shadow_train_loader = torch.utils.data.DataLoader(
    #         shadow_train, batch_size=args.batch_size, shuffle=False
    #     )
    #     shadow_test_loader = torch.utils.data.DataLoader(
    #         shadow_test, batch_size=args.batch_size, shuffle=False
    #     )

    #     target_train_loader = torch.utils.data.DataLoader(
    #         target_train, batch_size=args.batch_size, shuffle=False
    #     )
    #     target_test_loader = torch.utils.data.DataLoader(
    #         target_test, batch_size=args.batch_size, shuffle=False
    #     )

    #     evaluation_result["SVC_MIA_training_privacy"] = evaluation.SVC_MIA(
    #         shadow_train=shadow_train_loader,
    #         shadow_test=shadow_test_loader,
    #         target_train=target_train_loader,
    #         target_test=target_test_loader,
    #         model=model,
    #     )
    #     unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()

    # python main_evalmu.py --unlearn retrain --cp_path results/cifar10_resnet/origin/0model_SA_best.pth.tar --num_indexes_to_replace 4500 --unlearn_steps 182 --theta_lr 0.1 --w_path results/cifar10_resnet/worstmu_params/optm-retrain-num_4500-w_lr_10000.0/select_weight.pth.tar --save_dir results/cifar10_resnet/try
    # python main_evalmu.py --unlearn scrub --cp_path results/cifar10_resnet/origin/0model_SA_best.pth.tar --num_indexes_to_replace 4500 --unlearn_steps 10 --theta_lr 5e-4 --save_dir results/cifar10_resnet/try