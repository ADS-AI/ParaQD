import pandas as pd
import re
import numpy as np
import argparse
import os


def create_raw_versions(fpath, columns=["question"], dataset_name="AQUA", split="train"):
    csv = False
    if fpath.endswith(".csv"): csv = True
    elif fpath.endswith(".xlsx"): csv = False
    else: raise ValueError("File extension not recognized")

    if csv: df = pd.read_csv(fpath)
    else: df = pd.read_excel(fpath)

    raw_df = df.loc[:, columns]
    output_fname = os.path.join(os.getcwd(), "data", dataset_name, f"{dataset_name}_{split}_raw.csv")
    raw_df.to_csv(output_fname, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fpath", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="AQUA", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--columns", nargs="+", type=str, default=["question"], help="Columns to use")

    args = parser.parse_args()

    create_raw_versions(args.fpath, columns=args.columns, dataset_name=args.dataset, split=args.split)
