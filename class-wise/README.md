# Worst-case forget set on class-wise unlearning
This is the official repository for Worst-case forget set on class-wise unlearning. The code structure of this project is adapted from the [DP4TL](https://github.com/OPTML-Group/DP4TL) codebase.

## Requirements
You can install the necessary Python packages with:
```bash
pip install -r requirements.txt
```

We remark that to accelerate model training, this code repository is built based on [FFCV](https://arxiv.org/abs/2306.12517) and we refer its installation instructions to its [official website](https://ffcv.io/). In this work, we build our argument system via [fastargs](https://github.com/GuillaumeLeclerc/fastargs), and we provide a revised version [here](https://github.com/Phoveran/fastargs). The installation of the latest fastargs is automatically handled by the command above.

## Datasets
For ImageNet, we provide the preprocessed data (`.beton`) in this [link](https://drive.google.com/drive/folders/1o76KNQh8C9zXEPNhFgEtsmGfMj8gwT3e?usp=sharing). Please download the data and put them in the `data` folder. Replace the `train_path` and `val_path` of the `dataset` in the `.json` file within the configs with the path of the preprocessed data (`.beton`).

## Scripts
In this section, we provide the instructions to reproduce the results in our paper.

1. Get the origin model.
We first train the origin model (ResNet-18) on ImageNet using the following command:
    ```bash
    python src/experiment/imagenet_train_from_scratch.py --config-file configs/imagenet_train_from_scratch.json 
    ```

2. Replace the `model_path` of the `blo` in the `.json` file in the configs with the path of origin weight (`.ckpt`). Find the Worst-case forget set on class-wise unlearning. The selection weight will be saved at `file/experiments/selection_worst_case`.
    ```bash
    python src/experiment/class_wise_worst_case_mu.py --config-file configs/selection_worst_case.json --train.optimizer.lr 5e-5 --blo.w_lr 1e-4
    ```

3. Replace the `training` and `testing` of the `indices` in the `.json` file in the configs with the path of selection weight (`.indices`). Evaluate on the worst-case forget set.
    ```bash
    # Retrain
    python src/experiment/evaluation_retrain.py --config-file configs/evaluation_retrain.json --train.optimizer.lr {theta_lr}
    ```

    If you want to use approxiamte unlearning method(like FT, l1-sparse), you also should replace the `model_path` of the `logging` in the `.json` file in the configs with the path of origin weight (`.ckpt`).
    ```bash
    # l1-sparse
    python src/experiment/evaluation_ft.py --config-file configs/evaluation_retrain.json --train.optimizer.lr {theta_lr} --train.alpha {alpha}
    ```
    If `indices` are `null`, evaluation will be performed on the random forget set.