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
from tqdm import tqdm

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

    m = py.shape[0]
    trace = avg_trace * 2
    noise_matrix = np.zeros(shape=(m, 2, 2))
    for i, py_slice in enumerate(py):
        if not isinstance(trace, float):
            trace_i = trace[i]
        else:
            trace_i = trace
        noise_matrix[i] = generate_noise_matrix_from_trace(
            2,
            trace=trace_i,
            py=py_slice,
            valid_noise_matrix=True,
            seed=seed+i,
        )

    # Map labels to to unique label indices
    label_to_index = {tuple(label): i for i, label in enumerate(unique_labels)}
    y_train_index = np.array([label_to_index[tuple(label)] for label in y_train])

    # Generate our noisy labels using the noise_matrix for each class
    y_train_noisy = np.copy(y_train)
    y_train_noisy_T = y_train_noisy.T


    # Only flip 3 labels at most
    all_indices = np.arange(y_train_noisy.shape[0])
    for i, noise_matrix_slice in enumerate(noise_matrix):
        # Get indices of labels that are not already flipped more than 3 times
        indices = np.intersect1d(
            np.where(
                np.sum(y_train_noisy != y_train, axis=1) < 3
            )[0],
            all_indices
        )
        if len(indices) == 0:
            continue
        y_train_noisy_T[i][indices] = generate_noisy_labels(
            y_train_noisy_T[i][indices],
            noise_matrix_slice,
        )

        # remaining_indices = np.setdiff1d(remaining_indices, indices)
    y_train_noisy = y_train_noisy_T.T

    unique_noisy_labels = unique_labels
    if len(unique_labels) < 2**m and not np.all(
        np.array([tuple(label) in label_to_index for label in y_train_noisy])
    ):
        # Update the label_to_index
        unique_noisy_labels = np.unique(y_train_noisy, axis=0)
        # Get current number of unique labels
        n_unique_labels = len(label_to_index)
        for label in unique_noisy_labels:
            if tuple(label) not in label_to_index:
                label_to_index[tuple(label)] = n_unique_labels
                n_unique_labels += 1
    s_index =  np.array([label_to_index[tuple(label)] for label in y_train_noisy])
    ps = np.bincount(s_index) / float(len(s_index))
    index_to_label = {i: label for label, i in label_to_index.items()}
    s = np.array([index_to_label[i] for i in s_index])


    # pred_probs = get_pred_probs(seed, cv_n_folds, X_train, s)

    label_errors_mask = s_index != y_train_index
    multiple_errors_mask_dict = {
        i: np.sum(y_train != y_train_noisy, axis=1) > i
        for i in range(1, 4)
    }

    return {
        "X_train": X_train,
        "true_labels_train": y_train,
        "X_test": X_test,
        "true_labels_test": y_test,
        "labels": s,
        "label_errors_mask": label_errors_mask,
        "multiple_errors_mask_dict": multiple_errors_mask_dict,
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
    small_dataset_kwargs = dataset_kwargs["small"]
    large_dataset_kwargs = dataset_kwargs["large"]

    seeds = all_params["seeds"]

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for kwargs, dataset_name_prefix in zip(
        [small_dataset_kwargs, large_dataset_kwargs],
        ["small", "large"],
    ):
        gamma_params = kwargs.pop("gamma")
        shape, scale = gamma_params["shape"], gamma_params["scale"]
        for seed in tqdm(seeds):
            np.random.seed(seed)
            dataset_name = f"{dataset_name_prefix}_dataset_{str(seed).zfill(4)}"
            # Generate avg_trace per class from gamma distribution
            K=kwargs["n_classes"]
            avg_traces = 1 - np.random.gamma(shape, scale, size=K)
            avg_traces *= (1 - (np.exp(-np.arange(K)**2/K+np.log(1/(2*K)))))[np.argsort(avg_traces)][::-1]
            avg_traces = np.maximum(avg_traces, 1 - avg_traces)
            kwargs["avg_trace"] = avg_traces
            make_dataset(
                seed=seed,
                dataset_kwargs=kwargs,
                output_dir=OUTPUT_DIR,
                dataset_name=dataset_name,
            )
