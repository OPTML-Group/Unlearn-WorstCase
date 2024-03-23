import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import torchvision.transforms as transforms
import os

import utils
import arg_parser


def get_median(features, targets):
    # get the median feature vector of each class
    num_classes = len(np.unique(targets, axis=0))
    prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)
    
    for i in range(num_classes):
        prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
    return prot


def get_distance(features, labels):
    
    prots = get_median(features, labels)
    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))
    
    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        prots_for_each_example[(labels==i).nonzero()[0], :] = prots[i]
    distance = np.linalg.norm(features - prots_for_each_example, axis=1)
    
    return distance


def get_features(args):

    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    # obtain features of each sample
    (
        model, 
        train_set, 
        _, 
        _
    ) = utils.setup_model_dataset(args)

    model.cuda()
    checkpoint = torch.load(args.cp_path, map_location=device)["state_dict"]
    model.load_state_dict(checkpoint, strict=True)

    train_full_loader =  DataLoader(train_set, batch_size=256, pin_memory=True)
    
    targets, features = [], []
    for img, target in tqdm(train_full_loader):
        targets.extend(target.numpy().tolist())
        
        img = img.to(device)
        feature = model(img).detach().cpu().numpy()
        features.extend([feature[i] for i in range(feature.shape[0])])
    
    features = np.array(features)
    targets = np.array(targets)
    
    return features, targets

def get_save_idx(args, distance):
    
    k = args.num_indexes_to_replace  # Number of elements to select
    
    sorted_idx = distance.argsort()
    mid_idx = len(sorted_idx) // 2  # Find the middle index
    
    # Calculate the start and end indices of the slice
    start_idx = mid_idx - k // 2
    end_idx = start_idx + k
    
    # ids = sorted_idx[start_idx:end_idx]
    ids = np.concatenate((sorted_idx[:start_idx], sorted_idx[end_idx:]))
    
    return ids

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet50", help="backbone architecture")

    parser.add_argument(
        "--save_dir", 
        default="results/cifar10_resnet/coreset", 
        type=str, 
        help="dir to save saved image ids"
    )
    parser.add_argument("--rate", type=float, default=0.2, help="selection ratio")
    parser.add_argument("--cp_path", type=str, default=None, help="model")
    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=4500,
        help="Number of data to forget",
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    features, targets = get_features(args)
    distance = get_distance(features, targets)
    ids = get_save_idx(args, distance)
    print(ids, len(ids))
    os.makedirs(args.save_dir, exist_ok=True)

    if args.dataset == "TinyImagenet":
        temp_ids = np.zeros(100000)
    else:
        temp_ids = np.zeros(45000)
    temp_ids[ids] = 1

    w = {"w": [temp_ids]}
    w_path = os.path.join(args.save_dir, "select_weight.pth.tar")

    torch.save(w, w_path)

if __name__ == "__main__":
    main()