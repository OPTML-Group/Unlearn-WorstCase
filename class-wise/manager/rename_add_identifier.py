import argparse
import json
import os


def rename_directory(old_folder_path, new_folder_name, test=True):
    root = os.path.dirname(old_folder_path)
    new_folder_path = os.path.join(root, new_folder_name)
    if not test:
        os.rename(old_folder_path, new_folder_path)
    print(f"Renamed folder: {old_folder_path} -> {new_folder_path}")


if __name__ == '__main__old':
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, default="file/experiments/imagenet_transfer_to_downstream/")
    args = p.parse_args()

    assert os.path.isdir(args.path), 'exp_path should be path to a folder that contains all the experiments!'
    dataset_list = ['dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'ucf101', 'cifar10', 'cifar100', 'waterbirds', 'flowers102']

    num_list = range(50, 951, 50)

    for root, dirs, files in os.walk(args.path):
        for dir in dirs:
            if "2023" in dir:
                found_flag = False
                full_dir_path = os.path.join(root, dir)
                config_path = os.path.join(full_dir_path, "config.json")
                f = open(config_path, 'r')
                data = json.load(f)
                pretrained_path = data["network"]["pretrained_ckpt"]
                for dataset in dataset_list:
                    if dataset in pretrained_path:
                        for num in num_list:
                            if f"_{str(num)}_" in pretrained_path:
                                new_name = f"{dataset}_{str(num)}_rn101"
                                rename_directory(full_dir_path, new_name)
                                found_flag = True
                if not found_flag:
                    print(f"Found unidentified folder {full_dir_path} with pretrained checkpoint path: {pretrained_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, default="file/experiments/imagenet_transfer_to_downstream/")
    p.add_argument('--test', action="store_true", default=False)
    args = p.parse_args()

    assert os.path.isdir(args.path), 'exp_path should be path to a folder that contains all the experiments!'

    for root, dirs, files in os.walk(args.path):
        for dir in dirs:
            if "2023" in dir:  # no identifier
                full_dir_path = os.path.join(root, dir)
                config_path = os.path.join(full_dir_path, "config.json")
                f = open(config_path, 'r')
                data = json.load(f)

                assert data["network"]["architecture"] == "vit_b_16"
                if not data["dataset"]["prune"]:
                    new_name = "vit_b_16_full_pretrain"
                else:
                    indices_name = data["dataset"]["indices"]["training"]
                    if "flm" in indices_name:
                        dataset = indices_name.split("/")[-1].split("_flm")[0]
                        num = indices_name.split("top")[-1].split(".indices")[0]
                        new_name = f"{dataset}_vit_b_16_flm_{num}"
                    elif "random" in indices_name:
                        num = indices_name.split("random_")[-1].split(".indices")[0]
                        new_name = f"vit_b_16_random_{num}"
                    else:
                        raise ValueError

                rename_directory(full_dir_path, new_name, test=args.test)




