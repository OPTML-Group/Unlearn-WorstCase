import os
import torch

exps = ['2023-04-11-23-06-50-100100', '2023-04-11-23-58-40-263551', '2023-04-11-23-58-40-264033', '2023-04-11-23-58-40-264063', '2023-04-12-00-10-05-023294']

for epoch in range(5):
    score = torch.stack([torch.load(os.path.join('file/experiments/imagenet_train_from_scratch', exp, 'grad_norm_scores', f'epoch{epoch}.score'), map_location='cpu') for exp in exps]).mean(0)
    for k in range(50, 1000, 50):
        chosen_indices = torch.topk(score, int((k/1000)*score.size(0))).indices
        torch.save(chosen_indices, f'file/selectors/grad_norm_epoch{epoch}_train_{k}.indices')
