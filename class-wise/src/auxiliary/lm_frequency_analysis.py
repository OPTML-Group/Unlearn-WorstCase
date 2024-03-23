import torch
import numpy as np
from matplotlib import pyplot as plt


datasets = ["dtd", "flowers102", "oxfordpets", "sun397", "ucf101", "food101", "stanfordcars"]

for dataset in datasets:
    frequency = torch.load(f"file/frequency/{dataset}_flm.frequency", map_location="cpu")
    frequency, frequency_index = torch.sort(frequency)
    # frequency_at_point = frequency[frequency_index[int(len(frequency_index) * point)]]
    # print(f"{dataset}: {frequency_at_point}")
    # frequency_removed_sum = frequency[:int(len(frequency)*point)].sum() / frequency.sum()
    print(f"{dataset}: {len(frequency[frequency==0])}")