import pandas as pd
import re
import numpy as np


def create_raw_versions(fpath, columns=["question"], dataset_name="AQUA", split="train"):
    csv = False
    if fpath.endswith(".csv"): csv = True
    elif fpath.endswith(".xlsx"): csv = False
    else: raise ValueError("File extension not recognized")

    if csv: df = pd.read_csv(fpath)
    else: df = pd.read_excel(fpath)

    raw_df = df.loc[:, columns]
    output_fname = f"data/{dataset_name}_{split}_raw.csv"
    raw_df.to_csv(output_fname, index=False)



if __name__ == "__main__":
    create_raw_versions("D:/Material/NLP/NLP Projects/ParaQD\ParaQD/val_samples_augmented_aqua_21821_22K.csv",
                         dataset_name="AQUA", split="val")
