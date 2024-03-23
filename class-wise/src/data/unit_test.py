import tqdm
import shutil
import importlib
import os
import arguments.data

datasets = arguments.data.CPU_DATASETS


def unit_test_env(func):
    def _new_func(*args, **kwargs):
        if 'test_name' not in kwargs:
            name = None
        else:
            name = kwargs.pop('test_name')

        print(f"Unit test {name} start.")

        try:
            func(*args, **kwargs)
        except Exception as err:
            print(f"Get {err.__class__.__name__} on {name}:")
            print(err)
        else:
            print(f"Unit test success on {name}.")

    return _new_func


@unit_test_env
def gen_error():
    shutil.rmtree('asdfasdgasdgoi')


@unit_test_env
def gen_error_2():
    assert 1 == 2, "abcdefg"


@unit_test_env
def unit_test(dataset_module, temp_dir):
    batch_size = 256
    res = 224
    train_loader, _ = dataset_module.get_train_loader(
        temp_dir, 8, batch_size=batch_size, res=res)
    test_loader, _ = dataset_module.get_test_loader(
        temp_dir, 8, batch_size=batch_size, res=res)

    count = 0

    for img, label, idx in tqdm.tqdm(train_loader):
        if not (img.shape[0] == batch_size and label.shape[0] == batch_size and idx.shape[0] == batch_size):
            count += 1
        assert count < 2, "Batch size doesn't match for more than 1 batch"
        assert img.shape[-1] == res and img.shape[-2] == res, "Resolution not match"

    count = 0

    for img, label, idx in tqdm.tqdm(test_loader):
        if not (img.shape[0] == batch_size and label.shape[0] == batch_size and idx.shape[0] == batch_size):
            count += 1
        assert count < 2, "Batch size doesn't match for more than 1 batch"
        assert img.shape[-1] == res and img.shape[-2] == res, "Resolution not match"


def test_cpu_datasets(temp_dir='./temp'):
    for dataset_name in datasets:
        module = importlib.import_module(dataset_name, '.')
        unit_test(module, os.path.join(temp_dir, dataset_name),
                  test_name=f"dataset_{dataset_name}")
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_cpu_datasets()
    # gen_error(test_name="error")
    # gen_error_2(test_name="error")
