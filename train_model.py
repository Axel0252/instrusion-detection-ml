import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

import misc

# Configs
# DROP_COLUMNS = ["Src IP", "Dst IP", "Flow ID"] not dropped in the latest experiments
CLEAN_DATASET = "cic_ids2018_cleaned.csv"

# Argument parser
parser = argparse.ArgumentParser(prog="train_model")
parser.add_argument(
    "-y",
    "--hyp",
    type=str,
    default=None,
    help="(Optional) Set the specified class to choose the best hyperparameters (accepts only 'gs' or 'rs')",
)
args = parser.parse_args()


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
    if args.hyp is not None:
        df_small = df_equal.sample(200000, random_state=42)
        print("Shape df_small:", df_small.shape)
        X_train, X_test = train_test_split(df_small, test_size=0.3, random_state=42)
        y_train = X_train.pop("Label")
        y_test = X_test.pop("Label")

        model = find_best_hyp(X_train, y_train, args.hyp)
    else:
        X_train, X_test = train_test_split(df_equal, test_size=0.3, random_state=42)
        y_train = X_train.pop("Label")
        y_test = X_test.pop("Label")

        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=20,
            min_samples_leaf=1,
            max_features="sqrt",
            max_depth=None,
            n_jobs=-1,
        )

    print("\nTraining...")
    model.fit(X_train, y_train)

    return model, X_test, X_train, y_test


def find_best_hyp(X_train, y_train, alg):
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    if alg == "rs":
        # RandomizedSearchCV
        # Best params: {'n_estimators': 100, 'min_samples_split': 20, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
        # Best score: 0.9823069477615828
        hyperparameters = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 15, 25, 40],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2"],
        }
        cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=hyperparameters,
            n_iter=30,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
            scoring="f1",
        )
        cv.fit(X_train, y_train)
    elif alg == "gs":
        # GridSearchCV
        param_grid = {
            "n_estimators": [300],
            "max_depth": [None, 30],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
        }
        cv = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            verbose=1,
            n_jobs=-1,
            scoring="f1",
        )
        cv.fit(X=X_train, y=y_train)
    else:
        print("Wrong hyp algorithm selected.")
        return -1

    # Results
    print("Best params:\n", cv.best_params_)
    print("Best score:\n", cv.best_score_)
    best_model = cv.best_estimator_
    results = pd.DataFrame(cv.cv_results_)
    print(
        results[
            [
                "mean_test_score",
                "param_n_estimators",
                "param_min_samples_split",
                "param_max_depth",
                "param_min_samples_leaf",
                "param_max_features",
            ]
        ].sort_values(by="mean_test_score", ascending=False)
    )
    print("\nTraining...")
    best_model.fit(X=X_train, y=y_train)

    return best_model


def evaluate(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]

    # Default threshold
    y_pred = (y_proba >= 0.5).astype(int)

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

    # ROC CURVE
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 6))

    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    plt.savefig("imgs/roc_curve.png", dpi=300)

    # CONFUSION MATRIX
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
    if args.hyp is None:
        X_small = X_test.sample(500000, random_state=42)
        y_small = y_test[X_small.index]

    # result = permutation_importance(
    #     model, X_test, y_test, n_repeats=2, random_state=42, n_jobs=1
    # )

    # fi = pd.Series(result.importances_mean, index=X_test.columns).sort_values(
    #     ascending=False
    # )

    fn = X_test.columns
    fi = pd.DataFrame({"feature": fn, "importance": model.feature_importances_})

    fi = fi.sort_values(by="importance", ascending=False)
    top10 = fi.head(10)

    print(fi.head(10))

    plt.figure(figsize=(10, 6))

    plt.barh(top10["feature"], top10["importance"])

    plt.gca().invert_yaxis()

    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importances")

    plt.tight_layout()
    plt.savefig("imgs/feature_importance.png", dpi=300)

    return fi


def main():
    if not os.path.exists(CLEAN_DATASET):
        misc.build_dataset()

    df_equal = load_and_prepare()

    best_model, X_test, X_train, y_test = train_model(df_equal)

    evaluate(best_model, X_test, y_test)

    feature_importance(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
