import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

import misc

# Configs
DATASET_PATH = "dataset/"
CLEAN_DATASET = "cic_ids2018_cleaned.csv"
CHUNK_SIZE = 500000
DROP_COLUMNS = ["Timestamp", "Src IP", "Dst IP", "Flow ID"]


def load_and_prepare():
    df: pd.DataFrame = pd.read_csv(CLEAN_DATASET, on_bad_lines="skip")

    df1: pd.DataFrame = df.loc[df["Label"] == 0][:2169002]
    df2: pd.DataFrame = df.loc[df["Label"] == 1][:2169002]
    df_equal: pd.DataFrame = pd.concat([df1, df2], axis=0)
    misc.print_info(df_equal)

    df_equal.columns = df_equal.columns.str.strip()

    print("Shape df:", df_equal.shape)

    return df_equal


def train_model(df_equal: pd.DataFrame):
    X_train, X_test = train_test_split(df_equal, test_size=0.3, random_state=12)

    y_train = X_train.pop("Label")
    y_test = X_test.pop("Label")

    model = RandomForestClassifier(
        n_estimators=75,
        criterion="gini",
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    )

    # hyperparameters = {"n_estimators": [75]}

    # clf = GridSearchCV(
    #     estimator=model,
    #     param_grid=hyperparameters,
    #     cv=2,
    #     verbose=1,
    #     n_jobs=-1,
    # )

    print("\nTraining...")
    # clf.fit(X=X_train, y=y_train)
    model.fit(X=X_train, y=y_train)

    return model, X_test, X_train, y_test


def evaluate(model, X_test, y_test, threshold=0.43):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
    }

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

    cm = [[tn, fp], [fn, tp]]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig("imgs/confusion_matrix.png", dpi=300)

    return metrics


def feature_importance(model, X_test, y_test, top_n=10):
    X_small = X_test.sample(50000, random_state=42)
    y_small = y_test[X_small.index]

    result = permutation_importance(
        model, X_small, y_small, n_repeats=2, random_state=42, n_jobs=-1
    )

    fi = pd.Series(result.importances_mean, index=X_test.columns).sort_values(
        ascending=False
    )

    print(fi.head(top_n))

    plt.figure(figsize=(10, 6))

    fi.head(top_n).sort_values().plot(kind="barh")

    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importance")

    plt.tight_layout()
    plt.savefig("imgs/feature_importance.png", dpi=300)

    return fi


def main():
    if not os.path.exists(CLEAN_DATASET):
        misc.build_dataset()

    df_equal = load_and_prepare()

    model, X_test, X_train, y_test = train_model(df_equal)

    evaluate(model, X_test, y_test)

    feature_importance(model, X_test, y_test)


if __name__ == "__main__":
    main()
