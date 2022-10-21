"""
This script does the following:
- Load the multilabel datasets.
- Use cross-validation to get the predicted probabilities of a 1vR-classifier on the training set.
"""

import pickle
import yaml
import pathlib
import numpy as np
from tqdm import tqdm



from cleanlab.internal.util import train_val_split
import cleanlab.internal.multilabel_utils as mlutils
import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = pathlib.Path("data/generated")
OUTPUT_DIR = pathlib.Path("data/pred_probs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_split_generator(labels, cv):
    unique_labels = np.unique(labels, axis=0)
    label_to_index = {tuple(label): i for i, label in enumerate(unique_labels)}
    multilabel_ids = np.array([label_to_index[tuple(label)] for label in labels])
    split_generator = cv.split(X=multilabel_ids, y=multilabel_ids)
    return split_generator


def train_fold(X, labels, *, clf, pred_probs, cv_train_idx, cv_holdout_idx):
    clf_copy = sklearn.base.clone(clf)
    X_train_cv, X_holdout_cv, s_train_cv, _ = train_val_split(
        X, labels, cv_train_idx, cv_holdout_idx
    )
    clf_copy.fit(X_train_cv, s_train_cv)
    pred_probs[cv_holdout_idx] = clf_copy.predict_proba(X_holdout_cv)

def get_cross_validated_multilabel_pred_probs(X, labels, *, clf, cv):
    split_generator = get_split_generator(labels, cv)
    pred_probs = np.zeros(shape=labels.shape)
    for cv_train_idx, cv_holdout_idx in split_generator:
        train_fold(
            X,
            labels,
            clf=clf,
            pred_probs=pred_probs,
            cv_train_idx=cv_train_idx,
            cv_holdout_idx=cv_holdout_idx,
        )
    return pred_probs

def get_pred_probs(seed: int, cv_n_folds: int, X, s):
    clf = OneVsRestClassifier(LogisticRegression())
    kf = sklearn.model_selection.StratifiedKFold(
        n_splits=cv_n_folds,
        shuffle=True,
        random_state=seed,
    )
    pred_probs = get_cross_validated_multilabel_pred_probs(X, s, clf=clf, cv=kf)
    return pred_probs


if __name__ == "__main__":
    # Load params.yaml
    all_params = yaml.load(open("params.yaml"), Loader=yaml.FullLoader)
    train_params = all_params["train"]
    seed = train_params["seed"]
    cv_n_folds = train_params["cv_n_folds"]

    # Load the datasets
    dataset_files = list(DATA_DIR.glob("*.pkl"))
    dataset_files.sort()

    clf_log_reg = OneVsRestClassifier(LogisticRegression(random_state=seed), n_jobs=-1)
    clf_rf = OneVsRestClassifier(RandomForestClassifier(random_state=seed), n_jobs=-1)

    kf = sklearn.model_selection.StratifiedKFold(
        n_splits=cv_n_folds,
        shuffle=True,
        random_state=seed,
    )

    for dataset_file in tqdm(dataset_files):
        dataset = pickle.load(open(dataset_file, "rb"))
        # Unpack the dataset
        X_train, true_labels_train, X_test, true_labels_test, labels, label_errors_mask, multiple_errors_mask_dict, ps, py, noise_matrix, m, n = [
            dataset[k] for k in dataset.keys()
        ]

        pred_probs_dict = {}
        for clf_name, clf in [("log_reg", clf_log_reg), ("rf", clf_rf)]: 
            pred_probs = mlutils.get_cross_validated_multilabel_pred_probs(X_train, labels, clf=clf, cv=kf)
            pred_probs_dict[clf_name] = pred_probs

        # Save the predicted probabilities
        output_file = OUTPUT_DIR / dataset_file.name
        pickle.dump(pred_probs_dict, open(output_file, "wb"))
