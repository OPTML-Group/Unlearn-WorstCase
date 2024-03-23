# Worst-case forget set on prompt-wise unlearning
This is the official repository for Worst-case forget set on prompt-wise unlearning.

## Requirements
This repo is based on the environment `ip2p`.

Install the environment using the following command:
```bash
conda env create -f environment.yaml
```

## Download model and UnlearnCanvas dataset
1. [Model](https://drive.google.com/file/d/1F4cngFOKZYyebgLu4JoyKBhKmZrzr7ub/view?usp=drive_link)
2. [UnlearnCanvas dataset](https://huggingface.co/datasets/OPTML-Group/UnlearnCanvas)

## Scripts
1. Utilize **ckpt_path** and **data_path** to specify the model and dataset path. Employ ESD with SignSGD to identify the worst-case forget set on prompt-wise unlearning.
```bash
python stable_diffusion/train-scripts/select-pair.py --train_method=xattn --SignESD --lr 1e-5 --w_lr 100.0 --devices 0,0 --output_dir results/select-pair --ckpt_path {the path of origin model} --data_path {the path of dataset}
```

2. Employ ESD with AdamW to unlearn the worst-case forget set. Subsequently, utilize the unlearned model to generate associated images.
```bash
python stable_diffusion/train-scripts/evaluate_selection.py --train_method=xattn --lr 3e-7 --devices 0,0 --output_dir results/evaluation --ckpt_path {the path of origin model} --data_path {the path of dataset} --w_path {path of the selection weight}
```
If **random_choice**, the original model will unlearn on a random forget set.