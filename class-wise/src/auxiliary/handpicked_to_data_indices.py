import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser("Handpicked to Data Indices")
parser.add_argument("--hand-picked-file", '--hp', required=True, type=str)
parser.add_argument("--train-label-file", required=True, type=str)
parser.add_argument("--test-label-file", required=True, type=str)
parser.add_argument("--output-path", '--output', required=True, type=str)
args = parser.parse_args()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def tensor_a_in_b(a, b):
    a_expanded = a.unsqueeze(1).expand(-1, b.shape[0])
    b_expanded = b.unsqueeze(0).expand(a.shape[0], -1)

    matches = (a_expanded == b_expanded).any(dim=1)
    return matches


if __name__ == "__main__":
    hand_picked_class = torch.from_numpy(np.load(args.hand_picked_file))
    train_label_info = torch.load(args.train_label_file, map_location="cpu")
    test_label_info = torch.load(args.test_label_file, map_location="cpu")

    train_indices = train_label_info[tensor_a_in_b(train_label_info[:, 0], hand_picked_class), 1]
    test_indices = test_label_info[tensor_a_in_b(test_label_info[:, 0], hand_picked_class), 1]

    torch.save(train_indices, f"{args.output_path}_train.indices")
    torch.save(test_indices, f"{args.output_path}_test.indices")
