"""
This script does the following:
- Create 10 multilabel datasets with sklearn's dataset generator.
- Add noise to the labels.
- Save the datasets to disk.
"""


import pickle
from typing import Optional
import yaml
import pathlib
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

from cleanlab.benchmarking.noise_generation import (
    generate_noise_matrix_from_trace,
    generate_noisy_labels,
)

import cleanlab.internal.multilabel_utils as mlutils


OUTPUT_DIR = pathlib.Path("data/generated")

def generate_noisy_dataset(
    *,
    n_samples,
    n_features,
    n_classes,
    n_labels,
    length,
    allow_unlabeled,
    sparse,
    avg_trace,
    test_size,
    seed,
):
    """Generate toy dataset and add noise to the labels based on the label distribution in the training split."""
    X, y, *_ = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        length=length,
        allow_unlabeled=allow_unlabeled,
        sparse=sparse,
        return_indicator="dense",
        return_distributions=True,
    )

    # Normalize features
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Count unique labels
    unique_labels = np.unique(y, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    # Compute p(true_label=k)
    py = mlutils.multilabel_py(y_train)

    m = len(unique_labels)
    trace = avg_trace * m
    noise_matrix = generate_noise_matrix_from_trace(
        m,
        trace=trace,
        py=py,
        valid_noise_matrix=True,
        seed=seed,
    )

    # Map labels to to unique label indices
    label_to_index = {tuple(label): i for i, label in enumerate(unique_labels)}
    y_train_index = np.array([label_to_index[tuple(label)] for label in y_train])

    # Generate our noisy labels using the noise_matrix.
    s_index = generate_noisy_labels(y_train_index, noise_matrix)
    ps = np.bincount(s_index) / float(len(s_index))
    s = np.array([unique_labels[i] for i in s_index])

    # pred_probs = get_pred_probs(seed, cv_n_folds, X_train, s)

    label_errors_mask = s_index != y_train_index

    return {
        "X_train": X_train,
        "true_labels_train": y_train,
        "X_test": X_test,
        "true_labels_test": y_test,
        "labels": s,
        "label_errors_mask": label_errors_mask,
        "ps": ps,
        "py": py,
        "noise_matrix": noise_matrix,
        "m": m,
        "n": X.shape[0],
    }


def make_dataset(*, seed: int, dataset_kwargs: dict, output_dir: str, dataset_name: Optional[str] = None):
    dataset = generate_noisy_dataset(**{**dataset_kwargs, "seed": seed})
    # Pad seed to 4 digits
    if dataset_name is None:
        dataset_name = f"dataset_{str(seed).zfill(4)}"
    pickle_file = pathlib.Path(output_dir) / f"{dataset_name}.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(dataset, f)



if __name__ == "__main__":
    # Load params.yaml
    all_params = yaml.load(open("params.yaml"), Loader=yaml.FullLoader)
    # Get params for generate_noisy_dataset
    dataset_kwargs = all_params["dataset_kwargs"]
    seeds = all_params["seeds"]

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        dataset_name = f"dataset_{str(seed).zfill(4)}"
        make_dataset(
            seed=seed,
            dataset_kwargs=dataset_kwargs,
            output_dir=OUTPUT_DIR,
            dataset_name=dataset_name,
        )
