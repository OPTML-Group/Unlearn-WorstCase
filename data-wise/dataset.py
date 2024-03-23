"""
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
"""
import copy
import glob
import os
from shutil import move

import numpy as np
import torch
import pandas
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from tqdm import tqdm




def replace_indexes(
    dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        try:
            dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = -dataset.labels[indexes] - 1
            except:
                try:
                    dataset.attr[indexes] = -dataset.attr[indexes] - 1
                except:
                    dataset._labels[indexes] = -dataset._labels[indexes] - 1


def replace_class(
    dataset: torch.utils.data.Dataset,
    class_to_replace: int,
    num_indexes_to_replace: int = None,
    seed: int = 0,
    only_mark: bool = False,
):
    if class_to_replace == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:

                try:
                    indexes = np.flatnonzero(np.ones_like(dataset.attr))
                    # indexes = np.flatnonzero(np.ones_like(range(len(dataset.table))))
                except:
                    indexes = np.flatnonzero(np.ones_like(dataset._labels))

    else:
        try:
            indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        except:
            try:
                indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
            except:
                indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)

    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes
        ), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)
    return indexes


# CIFAR-10
def cifar10_datasets(
    data_dir="datasets/cifar10",
    seed: int = 1,
    no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    return train_set, valid_set, test_set


class Index_CIFAR10(CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = super(Index_CIFAR10, self).__getitem__(index)

        return img, target, index


def cifar10_index_datasets(
    data_dir="datasets/cifar10",
    seed: int = 1,
    no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = Index_CIFAR10(data_dir, train=True, transform=train_transform, download=True)

    test_set = Index_CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    return train_set, valid_set, test_set

# CIFAR-100
def cifar100_datasets(
    data_dir="datasets/cifar100",
    seed: int = 1,
    no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")
    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    return train_set, valid_set, test_set


class Index_CIFAR100(CIFAR100):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = super(Index_CIFAR100, self).__getitem__(index)

        return img, target, index


def cifar100_index_datasets(
    data_dir="datasets/cifar10",
    seed: int = 1,
    no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")
    train_set = Index_CIFAR100(data_dir, train=True, transform=train_transform, download=True)

    test_set = Index_CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    return train_set, valid_set, test_set


# TinyImageNet
class TinyImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform
        for sample in tqdm(image_folder_set.imgs[start:end]):
            self.targets.append(sample[1])
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.imgs[idx]), self.targets[idx]
        else:
            return self.imgs[idx], self.targets[idx]


class Index_TinyImageNetDataset(TinyImageNetDataset):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = super(Index_TinyImageNetDataset, self).__getitem__(index)

        return img, target, index


class TinyImageNet:
    """
    TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        self.tr_train = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = []

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join(args.data_dir, "train/")
        self.val_path = os.path.join(args.data_dir, "val/")
        self.test_path = os.path.join(args.data_dir, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    val_dict[split_line[0]] = split_line[1]

            paths = glob.glob(os.path.join(args.data_dir, "val/images/*"))
            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + "/images")
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + "/images")

            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + "/images/*")) < 25:
                    dest = self.val_path + str(folder) + "/images/" + str(file)
                else:
                    dest = self.test_path + str(folder) + "/images/" + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))

    def datasets(
           self,
           seed: int = 1
    ):
        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = TinyImageNetDataset(train_set, self.norm_layer)
        test_set = ImageFolder(self.test_path, transform=self.tr_test)
        test_set = TinyImageNetDataset(test_set, self.norm_layer)
        train_set.targets = np.array(train_set.targets)
        train_set.targets = np.array(train_set.targets)
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.imgs = train_set_copy.imgs[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.imgs = train_set_copy.imgs[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]

        return train_set, valid_set, test_set


    def index_datasets(
           self,
           seed: int = 1
    ):
        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = Index_TinyImageNetDataset(train_set, self.norm_layer)
        test_set = ImageFolder(self.test_path, transform=self.tr_test)
        test_set = Index_TinyImageNetDataset(test_set, self.norm_layer)
        train_set.targets = np.array(train_set.targets)
        train_set.targets = np.array(train_set.targets)
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.imgs = train_set_copy.imgs[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.imgs = train_set_copy.imgs[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]

        return train_set, valid_set, test_set


    def data_loaders(
        self,
        batch_size=128,
        data_dir="datasets/tiny",
        num_workers=2,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
    ):
        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = TinyImageNetDataset(train_set, self.norm_layer)
        test_set = ImageFolder(self.test_path, transform=self.tr_test)
        test_set = TinyImageNetDataset(test_set, self.norm_layer)
        train_set.targets = np.array(train_set.targets)
        train_set.targets = np.array(train_set.targets)
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.imgs = train_set_copy.imgs[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.imgs = train_set_copy.imgs[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]

        if class_to_replace is not None and indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        if class_to_replace is not None:
            replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )
            if num_indexes_to_replace is None or num_indexes_to_replace == 500:
                test_set.targets = np.array(test_set.targets)
                test_set.imgs = test_set.imgs[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
                test_set.targets = test_set.targets.tolist()
        if indexes_to_replace is not None:
            replace_indexes(
                dataset=train_set,
                indexes=indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )

        loader_args = {"num_workers": 0, "pin_memory": False}

        def _init_fn(worker_id):
            np.random.seed(int(seed))

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        val_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader
        
# CelebA
class CelebA(Dataset):
    def __init__(self, root_dir, target_attrs="Brown_Hair", domain_attrs=None, img_transform=transforms.ToTensor(), type="train") -> None:
        super().__init__()
        self.type = type
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.table = self.__load_table("Blond_NonBlond.csv")

        self.target_attrs = target_attrs
        self.domain_attrs = domain_attrs
        self.img_transform = img_transform

        self.filenames = self.table['file_name'].to_list()
        self.labels = self.table[target_attrs].to_numpy()
        if self.domain_attrs is not None:
            self.domains = self.table[self.domain_attrs].to_numpy()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.filenames[index]))
        if self.img_transform is not None:
            img = self.img_transform(img)
    
        label = self.labels[index]
        if self.domain_attrs is not None:
            domain = self.domains[index]
            return img, (label, domain)
        else:
            return img, label

    def __load_table(self, path):
        whole_table = pandas.read_csv(path).sample(frac=1).reset_index(drop=True)
        train_point = 36000
        val_point = 38000
        if self.type == "train":
            return whole_table.iloc[:train_point]
        elif self.type == "val":
            return whole_table.iloc[train_point:val_point]
        elif self.type == "test":
            return whole_table.iloc[val_point:]
        else:
            raise ValueError("Invalid dataset type!")


class Index_CelebA(CelebA):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, labels = super(Index_CelebA, self).__getitem__(index)

        return img, labels, index


def celeba_datasets(
    data_dir="data/celeba",
    seed: int = 1,
    no_aug=False,
):
    train_transform = transforms.Compose([
        transforms.CenterCrop((178, 178)),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop((178, 178)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),    
    ])

    train_set = CelebA(data_dir, img_transform=train_transform, type='train')
    valid_set = CelebA(data_dir, img_transform=test_transform, type='val')
    test_set = CelebA(data_dir, img_transform=test_transform, type='test')

    return train_set, valid_set, test_set


def celeba_index_datasets(
    data_dir="data/celeba",
    seed: int = 1,
    no_aug=False,
):

    train_transform = transforms.Compose([
        transforms.CenterCrop((178, 178)),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop((178, 178)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),    
    ])

    train_set = Index_CelebA(data_dir, img_transform=train_transform, type='train')
    valid_set = Index_CelebA(data_dir, img_transform=test_transform, type='val')
    test_set = Index_CelebA(data_dir, img_transform=test_transform, type='test')

    return train_set, valid_set, test_set