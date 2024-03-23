from torch.utils.data import DataLoader
import torchvision

class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)
        return x, y, index
    
def get_train_loader(path, num_workers, batch_size, res, shuffle=True, in_memory=False, augments=True):
    bigger_resolution = int(res*256/224)
    augments = [
            torchvision.transforms.Resize((bigger_resolution, bigger_resolution)),
            torchvision.transforms.RandomCrop((res, res)),
            torchvision.transforms.RandomHorizontalFlip(),
            ] if augments else [torchvision.transforms.Resize((res, res))]
    train_transform = torchvision.transforms.Compose(augments + [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])
    train_data = CIFAR100(root = path, train = True, download = True, transform = train_transform)
    return DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=num_workers, pin_memory=in_memory), None


def get_test_loader(path, num_workers, batch_size, res, in_memory=False):
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((res, res)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])
    test_data = CIFAR100(root = path, train = False, download = False, transform = test_transform)
    return DataLoader(test_data, batch_size, shuffle = False, num_workers=num_workers, pin_memory=in_memory), None