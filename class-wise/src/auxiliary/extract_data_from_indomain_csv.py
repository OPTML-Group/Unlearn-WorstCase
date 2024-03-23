import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm


def process_csv(input_file, output_prefix):
    if not os.path.exists(input_file):
        raise ValueError(f"File {input_file} does not exist.")

    if not input_file.lower().endswith(".csv"):
        raise ValueError("Input file must be a CSV file.")

    df = pd.read_csv(input_file)
    first_col = df.columns[0]
    third_col = df.columns[2]

    df_ranked = df.sort_values(by=third_col, ascending=False)
    indices = df_ranked[first_col].tolist()

    for ratio in tqdm(range(5, 100, 5)):
        top_n = int(len(indices) * (ratio / 100))
        selected_indices = indices[:top_n]

        output_path = f"{output_prefix}_Top{ratio}0.indices"
        torch.save(selected_indices, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CSV file and save data indices according to specified ratios.")
    parser.add_argument("--input", help="Path to the input CSV file.")
    parser.add_argument("--output", help="Prefix for the saved output files.")

    args = parser.parse_args()
    process_csv(args.input, args.output)
