"""
This script does the following:
- Load the multilabel datasets.
- Load the untrained classifier dictionary and cross-validator.
- Use cross-validation to get the predicted probabilities of a 1vR-classifier on the training set.
"""

import pickle
import yaml
import pathlib
import numpy as np
from tqdm import tqdm


import cleanlab.internal.multilabel_scorer as mlscorer
import sklearn
from sklearn.multiclass import OneVsRestClassifier

DATA_DIR = pathlib.Path("data/generated")
OUTPUT_DIR = pathlib.Path("data/pred_probs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main(
    dataset_files: list,
    clf_dict: dict,
    cross_validator: sklearn.model_selection.BaseCrossValidator,
) -> dict:
    """Compute the cross-validated predicted probabilities of a 1vR-classifier
    on the training split of each dataset and save them to disk.
    """
    for dataset_file in tqdm(dataset_files):
        dataset = pickle.load(open(dataset_file, "rb"))
        # Unpack the dataset
        X_train, labels = dataset["X_train"], dataset["labels"]
        pred_probs_dict = compute_pred_probs(clf_dict, cross_validator, X_train, labels)

        # Save the predicted probabilities
        output_file = OUTPUT_DIR / dataset_file.name
        pickle.dump(pred_probs_dict, open(output_file, "wb"))


def compute_pred_probs(
    clf_dict: dict[str, OneVsRestClassifier],
    cross_validator: sklearn.model_selection.BaseCrossValidator,
    X_train: np.ndarray,
    labels: np.ndarray,
) -> dict:
    pred_probs_dict = {}
    for clf_name, clf in clf_dict.items():
        pred_probs_dict[clf_name] = mlscorer.get_cross_validated_multilabel_pred_probs(
            X_train, labels, clf=clf, cv=cross_validator
        )
    return pred_probs_dict


if __name__ == "__main__":
    # Load the datasets
    dataset_files = list(DATA_DIR.glob("*.pkl"))
    dataset_files.sort()

    # Load params.yaml, no dependence on hyperparameters
    all_params = yaml.load(open("params.yaml"), Loader=yaml.FullLoader)
    train_params = all_params["train"]
    clf_dict_file = pathlib.Path(train_params["clf_dict"])
    cross_validator_file = pathlib.Path(train_params["cross_validator"])

    # Load the classifiers and cross-validator
    clf_dict = pickle.load(open(clf_dict_file, "rb"))
    cross_validator = pickle.load(open(cross_validator_file, "rb"))

    # Run the script
    main(
        dataset_files=dataset_files,
        clf_dict=clf_dict,
        cross_validator=cross_validator,
    )
