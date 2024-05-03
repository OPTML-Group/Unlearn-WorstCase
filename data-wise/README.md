# Worst-case forget set on data-wise unlearning
This is the official repository for Worst-case forget set on data-wise unlearning. The code structure of this project is adapted from the [Sparse Unlearn](https://github.com/OPTML-Group/Unlearn-Sparse) codebase.

## Requirements
```bash
pip install -r requirements.txt
```

## Scripts
1. Get the origin model.
    ```bash
    python main_train.py --arch {model name} --dataset {dataset name} --epochs {the number of epochs} --lr {learning rate} --save_dir {file to save the orgin model}
    ```

2. Find the worst-case forget set on data-wise unlearning.
    ```bash
    python main_selmu.py --arch {model name} --dataset {dataset name} --cp_path {path of the origin model} --select_epochs {upper level epochs(selection epochs)}--num_indexes_to_replace {data amount of forget set} --unlearn w_{unlearning method for lower level} --unlearn_steps {lower level steps(unlearning steps)} --theta_lr {lower level learning rate} --save_dir {file to save the unleanred model and selecition weight}
    ```
    A simple example for ResNet-18 on CIFAR-10, using FT as lower level unlearning method.
    ```bash
    python main_selmu.py --arch resnet18 --dataset cifar10 --cp_path results/cifar10_resnet/origin/0model_SA_best.pth.tar --select_epochs 2 --num_indexes_to_replace 4500 --unlearn w_FT --unlearn_steps 2 --theta_lr 1e-3 --save_dir results/cifar10_resnet/data-wise
    ```

3. Evaluate on the worst-case forget set
    ```bash
    python main_evalmu.py --arch {model name} --dataset {dataset name} --cp_path {path of the origin model} --unlearn {unlearning method for evaluation}  --num_indexes_to_replace {data amount of forget set} --unlearn_steps {evaluation unlearning steps} --theta_lr {lower level learning rate} --w_path {path of the selection weight} --save_dir {file to save the evaluation result}
    ```
    A simple example for ResNet-18 on CIFAR-10, using Retrain as evaluation unlearning method.
    ```bash
    python main_evalmu.py --arch resnet18 --dataset cifar10 --cp_path results/cifar10_resnet/origin/0model_SA_best.pth.tar --unlearn retrain --num_indexes_to_replace 4500 --unlearn_steps 182 --theta_lr 0.1 --w_path results/cifar10_resnet/data-wise/select_weight.pth.tar --save_dir results/cifar10_resnet/evaluation
    ```
    If **w_path** is not specified, the origin model will unlearn on random forget set.