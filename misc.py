import glob
import hashlib
import os

import numpy as np
import pandas as pd


def build_dataset():
    files = glob.glob(os.path.join("dataset/", "*.csv"))

    print(f"Found {len(files)} files\n")

    first = True

    for file in files:
        print(f"Processing: {file}")

        for chunk in pd.read_csv(
            file, chunksize=100000, engine="python", on_bad_lines="skip"
        ):
            chunk.to_csv("cic_ids2018_cleaned.csv", mode="a", header=first, index=False)
            first = False


def clean_dataset(name: str):
    df = pd.read_csv(name)
    print("======== DATASET INFO ========")
    # df.info()
    print("==============================")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("======== DATASET DESCRIBE ========")
    # print(df.describe())
    print("==============================")
    print("======== DATASET CLEANED ========")
    df.dropna(inplace=True)
    # remove later
    df.drop(["Timestamp"], axis=1, inplace=True)
    print("NaN:", df.isna().sum().sum())
    print("Inf:", np.isinf(df.to_numpy()).sum())
    print("Max value:", df.max().max())

    print("==============================")
    print("======== DATASET VALUES ========")
    # print(df["Label"].value_counts())
    print("==============================")
    print("===============CHANGING LABELS ===============")
    # df["Label"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)

    # mask = ~(df.astype(str) == df.columns).all(axis=1)

    # df = df[mask]
    print("=============== SAVING ===============")
    df.to_csv(name, index=False)
    print_info(df)

    return df


def row_hash(row):
    return hashlib.md5("||".join(map(str, row.values)).encode()).hexdigest()


def process_df():
    print("======= PROCESSING DF =======")

    seen = set()
    first = True

    temp_output = "dedup_temp.csv"

    if os.path.exists(temp_output):
        os.remove(temp_output)

    for chunk in pd.read_csv(
        "cic_ids2018_cleaned.csv",
        chunksize=500000,
        engine="python",
        on_bad_lines="skip",
    ):
        hashes = chunk.apply(row_hash, axis=1)

        mask = ~hashes.isin(seen)

        if not mask.any():
            continue

        seen.update(hashes[mask])

        chunk.loc[mask].to_csv(temp_output, mode="a", header=first, index=False)

        first = False

    print(df["Label"].value_counts())

    print("=============================")


def print_info(df: pd.DataFrame):
    print("======== DATASET INFO ========")
    print("Shape:", df.shape)
    df.info()
    print("==============================")


if __name__ == "__main__":
    if not os.path.exists("cic_ids2018_cleaned.csv"):
        build_dataset()
        df = clean_dataset("cic_ids2018_cleaned.csv")
        process_df()
    else:
        clean_dataset("cic_ids2018_cleaned.csv")
        # df = pd.read_csv("cic_ids2018_cleaned.csv", on_bad_lines="skip")

        # print_info(df)
