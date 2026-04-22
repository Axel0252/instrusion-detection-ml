import pandas as pd
import numpy as np
import glob
import os

from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Configs
DATASET_PATH = "dataset/"
CLEAN_DATASET = "cic_ids2018_cleaned.csv"
CHUNK_SIZE = 100000
# SAMPLE_PER_CHUNK = 5000 if pc slows down too much

DROP_COLUMNS = ['Timestamp', 'Src IP', 'Dst IP', 'Flow ID']

def clean_chunk(chunk):
    chunk.columns = chunk.columns.str.strip()

    if "Label" in chunk.columns:
        chunk = chunk[chunk["Label"] != "Label"]
    else:
        return None

    chunk.drop(columns=[c for c in DROP_COLUMNS if c in chunk.columns], inplace=True)

    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

    y = chunk["Label"]
    X = chunk.drop("Label", axis=1)

    X = X.apply(pd.to_numeric, errors='coerce')

    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    clean = X.copy()
    clean["Label"] = y

    return clean


def build_dataset():
    files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))

    first = True

    print(f"Found {len(files)} files\n")

    for file in files:
        print(f"Processing: {file}")

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            engine="python",
            on_bad_lines="skip"
        ):
            cleaned = clean_chunk(chunk)

            if cleaned is None or len(cleaned) == 0:
                continue

            df_attack = cleaned[cleaned["Label"] != "Benign"]
            df_benign = cleaned[cleaned["Label"] == "Benign"]
            if len(df_attack) == 0 or len(df_benign) == 0:
                continue

            # downsample benign traffic
            df_benign_down = resample(
                df_benign,
                replace=False,
                n_samples=min(len(df_benign), len(df_attack) * 2),
                random_state=42
            )

            cleaned = pd.concat([df_attack, df_benign_down])
            cleaned.to_csv(
                CLEAN_DATASET,
                mode="a",
                header=first,
                index=False
            )

            first = False

    print("\nDataset created")


def load_and_prepare():
    df = pd.read_csv(CLEAN_DATASET, on_bad_lines="skip")

    df.columns = df.columns.str.strip()

    X = df.drop("Label", axis=1)
    y = df["Label"]

    y = y.apply(lambda x: 0 if x == "Benign" else 1)

    X = X.replace([np.inf, -np.inf], np.nan)

    mask = X.notnull().all(axis=1)
    X = X[mask]
    y = y[mask]

    X = X.clip(-1e10, 1e10)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print("Shape X:", X.shape)
    # remove later
    print("NaN:", X.isna().sum().sum())
    print("Inf:", np.isinf(X.to_numpy()).sum())
    print("Max value:", X.max().max())

    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0:1, 1:2},
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining...")
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold = 0.43 # experimenting
    y_pred = (y_proba > threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("\n=== METRICS ===")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nTarget metrics (ATTACK = 1):")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))


def feature_importance(model, X):
    importances = model.feature_importances_

    fi = pd.Series(importances, index=X.columns)
    fi = fi.sort_values(ascending=False)

    print("\n=== TOP 10 FEATURE ===")
    print(fi.head(10))


def main():
    if not os.path.exists(CLEAN_DATASET):
        build_dataset()

    X, y = load_and_prepare()

    model, X_test, y_test = train_model(X, y)

    print("\nCross-validation (F1)...")
    X_sample = X.sample(50000, random_state=42)
    y_sample = y.loc[X_sample.index]
    scores = cross_val_score(model, X_sample, y_sample, cv=3, scoring="f1", n_jobs=-1)
    print("Average F1:", scores.mean())

    evaluate(model, X_test, y_test)

    feature_importance(model, X)


if __name__ == "__main__":
    main()
