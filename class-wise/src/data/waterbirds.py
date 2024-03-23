import os
import json

import torch
from PIL import Image
import torchvision
from torch.utils.data import DataLoader

waterbirds_mean = [0.485, 0.456, 0.406]
waterbirds_var = [0.229, 0.224, 0.225]


class Waterbirds(torch.utils.data.Dataset):
    def __init__(self,
                 root: str = './datasets/waterbirds',
                 train: bool = False,
                 download: bool = False,
                 transform=None):

        if train:
            split = "train"
        else:
            split = "test"

        json_path = os.path.join(root, f'{split}.json')
        if download:
            if not os.path.isdir(root):
                os.makedirs(root)
            tar_path = os.path.join(
                root, 'waterbird_complete95_forest2water2.tar.gz')
            if not os.path.exists(json_path) or not os.path.exists(tar_path):
                Waterbirds.download(root)

        self.root = root
        self.image_paths, self.targets = Waterbirds.load_json(json_path)

        self.length = len(self.targets)

        if transform is not None:
            self.preprocessing = transform
        else:
            self.preprocessing = lambda x: x

    @staticmethod
    def load_json(json_path: str):
        '''
            Read the json file that stores the image path and the label.
        '''
        image_paths, targets = [], []

        with open(json_path, 'r') as f:

            for line in f:
                example = json.loads(line)
                targets.append(example['y'])
                image_paths.append(example['x'])

        targets = torch.tensor(targets)

        return image_paths, targets

    @staticmethod
    def download(root: str):
        '''
            Download the Waterbirds dataset and the split information.
            The original dataset assumes a perfect validation split (each group
            has equal amount of annotations).
            We combine the train and valid data and perform a random split to
            get the training and validation data.
        '''
        os.system(
            "echo 'Downloading Waterbirds'\n"
            f"cd {root}\n"
            "wget https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz\n"
            "tar -xf waterbird_complete95_forest2water2.tar.gz\n"
            "wget https://people.csail.mit.edu/yujia/files/ls/waterbirds/train.json\n"
            "wget https://people.csail.mit.edu/yujia/files/ls/waterbirds/valid.json\n"
            "wget https://people.csail.mit.edu/yujia/files/ls/waterbirds/test.json\n"
            "mv waterbird_complete95_forest2water2/* ./\n"
            "rm -rf waterbird_complete95_forest2water2\n"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
            Given the index, return the bird image and the binary label.
        '''
        target = self.targets[idx]

        with Image.open(os.path.join(self.root, self.image_paths[idx])) as raw:
            img = self.preprocessing(raw)

        return img, target, idx


def get_train_loader(path, num_workers, batch_size, res, shuffle=True, in_memory=False, augments=True):
    bigger_resolution = int(res * 256 / 224)
    augments = [
        torchvision.transforms.Resize((bigger_resolution, bigger_resolution)),
        torchvision.transforms.RandomCrop((res, res)),
        torchvision.transforms.RandomHorizontalFlip(),
    ] if augments else [torchvision.transforms.Resize((res, res))]
    train_transform = torchvision.transforms.Compose(augments + [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=waterbirds_mean, std=waterbirds_var),
    ])
    train_data = Waterbirds(root=path, train=True,
                            download=True, transform=train_transform)
    return DataLoader(train_data, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=in_memory), None


def get_test_loader(path, num_workers, batch_size, res, in_memory=False):
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((res, res)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=waterbirds_mean, std=waterbirds_var),
    ])
    test_data = Waterbirds(root=path, train=False,
                           download=False, transform=test_transform)
    return DataLoader(test_data, batch_size, shuffle=False, num_workers=num_workers, pin_memory=in_memory), None
