"""
This script does the following:
- Load the multilabel datasets.
- Train 1vR-classifiers on the noisy and clean training set.
- Evaluate the accuracy of the classifiers on the noisy and clean test set.
- Summarize the results with the average accuracy per classifier, per group of datasets.
"""
import pickle
import yaml
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score


DATA_DIR = pathlib.Path("data/generated")
OUTPUT_DIR = pathlib.Path("data/accuracy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "results.csv"

def train_on_dataset(
    clf_log_reg,
    clf_rf,
    df,
    dataset_file,
    X_train,
    labels,
    true_labels_train,
    X_test,
    noisy_test_labels,
    true_labels_test,
    label_to_id,
    labels_idx,
    true_labels_train_idx,
    noisy_test_labels_idx,
    true_labels_test_idx):
    row = {
        "dataset": dataset_file.stem,
        "dataset_group": dataset_file.stem.split("_")[0],
    }
    for clf, clf_str in zip(
        [clf_log_reg, clf_rf],
        ["Logistic regression", "Random forest"]
    ):
        # Add clf to row
        row["clf"] = clf_str
        for train_labels, train_str in zip(
            [labels, true_labels_train],
            
            ["Noisy train", "True train"]
        ):
            # Add train set to row
            row["train_set"] = train_str
            for test_labels, test_idx, test_str in zip(
                [noisy_test_labels, true_labels_test],
                [noisy_test_labels_idx, true_labels_test_idx],
                ["Noisy test", "True test"]
            ):
                # Add test set to row
                row["test_set"] = test_str
                clf.fit(X_train, train_labels)
                y_pred = clf.predict(X_test)
                # OOD get -1
                # y_pred_idx = np.array([label_to_id.get(tuple(label), -1) for label in y_pred])
                acc = accuracy_score(test_labels, y_pred)
                row["Accuracy"] = acc

                hamming_loss_ = hamming_loss(test_labels, y_pred)
                row["Hamming loss"] = hamming_loss_

                jaccard_score_ = jaccard_score(test_labels, y_pred, average="samples", zero_division=1)
                row["Jaccard score"] = jaccard_score_

                # Average acccuracy per class
                average_accuracy_ = np.mean(np.mean(test_labels == y_pred, axis=0))
                row["Average accuracy"] = average_accuracy_

                df = pd.concat([df, pd.DataFrame([row])], axis=0)
    return df

def preprocess_labels(labels, true_labels_train, noisy_test_labels, true_labels_test):
    combined_labels = np.concatenate([labels, true_labels_train, noisy_test_labels, true_labels_test], axis=0)
    unique_labels = np.unique(combined_labels, axis=0)
    label_to_id = {tuple(label): i for i, label in enumerate(unique_labels)}
    labels_idx = np.array([label_to_id[tuple(label)] for label in labels])
    true_labels_train_idx = np.array([label_to_id[tuple(label)] for label in true_labels_train])
    noisy_test_labels_idx = np.array([label_to_id[tuple(label)] for label in noisy_test_labels])
    true_labels_test_idx = np.array([label_to_id[tuple(label)] for label in true_labels_test])
    return label_to_id,labels_idx,true_labels_train_idx,noisy_test_labels_idx,true_labels_test_idx

def unpack_dataset(dataset):
    X_train, labels, true_labels_train, X_test, noisy_test_labels, true_labels_test, label_errors_mask, test_label_errors_mask  = [
        dataset["X_train"],
        dataset["labels"],
        dataset["true_labels_train"],
        dataset["X_test"],
        dataset["noisy_test_labels"],
        dataset["true_labels_test"],
        dataset["label_errors_mask"],
        dataset["test_label_errors_mask"],
    ]
    return X_train,labels,true_labels_train,X_test,noisy_test_labels,true_labels_test

if __name__ == "__main__":
    # Load params.yaml
    all_params = yaml.load(open("params.yaml"), Loader=yaml.FullLoader)
    train_params = all_params["train"]
    seed = train_params["seed"]

    # Load the datasets
    dataset_files = list(DATA_DIR.glob("*.pkl"))
    dataset_files.sort()


    clf_log_reg = OneVsRestClassifier(LogisticRegression(random_state=seed), n_jobs=-1)
    clf_rf = OneVsRestClassifier(RandomForestClassifier(random_state=seed), n_jobs=-1)

    COLUMNS = [
        "dataset",
        "dataset_group",
        "clf",
        "train_set",
        "test_set",
        "Accuracy",
        "Average accuracy",
        "Hamming loss",
        "Jaccard score",
    ]
    df = pd.DataFrame(columns=COLUMNS)

    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)

    for dataset_file in tqdm(dataset_files):
        if dataset_file.stem in df["dataset"].values:
            continue
        dataset = pickle.load(open(dataset_file, "rb"))
        X_train, labels, true_labels_train, X_test, noisy_test_labels, true_labels_test = unpack_dataset(dataset)

        label_to_id, labels_idx, true_labels_train_idx, noisy_test_labels_idx, true_labels_test_idx = preprocess_labels(labels, true_labels_train, noisy_test_labels, true_labels_test)
        # Add dataset/group to df
        df = train_on_dataset(clf_log_reg, clf_rf, df, dataset_file, X_train, labels, true_labels_train, X_test, noisy_test_labels, true_labels_test, label_to_id, labels_idx, true_labels_train_idx, noisy_test_labels_idx, true_labels_test_idx)
    
        df.to_csv(RESULTS_FILE, index=False)
