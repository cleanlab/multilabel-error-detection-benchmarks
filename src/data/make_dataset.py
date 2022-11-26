"""
This script does the following:
- Create 10 multilabel datasets with sklearn's dataset generator.
- Add noise to the labels.
- Save the datasets to disk.
"""


import pickle
from typing import Optional, Union
import yaml
import pathlib
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from cleanlab.benchmarking.noise_generation import (
    generate_noise_matrix_from_trace,
    generate_noisy_labels,
)

from cleanlab.internal.multilabel_scorer import multilabel_py


OUTPUT_DIR = pathlib.Path("data/generated")
IMAGE_DIR = pathlib.Path("data/images/generated")


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
    """Generate toy dataset and add noise to the labels
    based on the label distribution in the training split.
    """
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
    py = multilabel_py(y_train)

    noise_array = create_noise_array(
        seed, py, avg_trace
    )  # 2x2 noise matrices for K classes in a (K, 2, 2) array

    # Map labels to to unique label indices
    label_to_index = {tuple(label): i for i, label in enumerate(unique_labels)}
    y_train_index = np.array([label_to_index[tuple(label)] for label in y_train])
    y_test_index = np.array([label_to_index[tuple(label)] for label in y_test])

    # Generate our noisy labels using the noise_matrix for each class
    y_train_noisy = create_noisy_labels(y_train, noise_array)
    y_test_noisy = create_noisy_labels(y_test, noise_array)

    m = py.shape[0]
    unique_noisy_labels = unique_labels
    if len(unique_labels) < 2 ** m and not np.all(
        np.array(
            [
                tuple(label) in label_to_index
                for label in [*y_train_noisy, *y_test_noisy]
            ]
        )
    ):
        # Update the label_to_index
        unique_noisy_labels = np.unique(
            np.concatenate(
                [
                    unique_labels,  # Rare true labels could get removed, never seen in practice
                    y_train_noisy,  # Rare noisy labels could get generated
                    y_test_noisy,  # Test-specific noisy labels could get generated
                ],
                axis=0,
            ),
            axis=0,
        )
        # Get current number of unique labels
        n_unique_labels = len(label_to_index)
        for label in unique_noisy_labels:
            if tuple(label) not in label_to_index:
                label_to_index[tuple(label)] = n_unique_labels
                n_unique_labels += 1
    s_index = np.array([label_to_index[tuple(label)] for label in y_train_noisy])
    s_test_index = np.array([label_to_index[tuple(label)] for label in y_test_noisy])
    ps = np.bincount(s_index) / float(len(s_index))
    index_to_label = {i: label for label, i in label_to_index.items()}
    s = np.array([index_to_label[i] for i in s_index])

    label_errors_mask = s_index != y_train_index
    test_label_errors_mask = s_test_index != y_test_index
    multiple_errors_mask_dict = {
        i: np.sum(y_train != y_train_noisy, axis=1) > i for i in range(1, 4)
    }
    test_multiple_errors_mask_dict = {
        i: np.sum(y_test != y_test_noisy, axis=1) > i for i in range(1, 4)
    }

    return {
        "X_train": X_train,
        "true_labels_train": y_train,
        "X_test": X_test,
        "true_labels_test": y_test,
        "labels": s,
        "noisy_test_labels": y_test_noisy,
        "label_errors_mask": label_errors_mask,
        "test_label_errors_mask": test_label_errors_mask,
        "multiple_errors_mask_dict": multiple_errors_mask_dict,
        "test_multiple_errors_mask_dict": test_multiple_errors_mask_dict,
        "ps": ps,
        "py": py,
        "noise_matrix": noise_array,
        "m": m,
        "n": X.shape[0],
    }


def create_noisy_labels(y_true, noise_array):
    y_noisy = np.copy(y_true)
    y_noisy_T = y_noisy.T

    # Only flip 3 labels at most
    all_indices = np.arange(y_noisy.shape[0])
    for i, noise_matrix_slice in enumerate(noise_array):
        # Get indices of labels that are not already flipped more than 3 times
        indices = np.intersect1d(
            np.where(np.sum(y_noisy != y_true, axis=1) < 3)[0], all_indices
        )
        if len(indices) == 0:
            continue
        y_noisy_T[i][indices] = generate_noisy_labels(
            y_noisy_T[i][indices],
            noise_matrix_slice,
        )

    y_noisy = y_noisy_T.T
    return y_noisy


def create_noise_array(seed: int, py: np.ndarray, avg_trace: Union[float, np.ndarray]):
    m = py.shape[0]
    avg_traces = np.full(m, avg_trace) if isinstance(avg_trace, float) else avg_trace
    noise_array = np.zeros(shape=(m, 2, 2))
    for i, (py_slice, avg_trace) in enumerate(zip(py, avg_traces)):
        trace = 2 * avg_trace
        noise_array[i] = generate_noise_matrix_from_trace(
            2,
            trace=trace,
            py=py_slice,
            valid_noise_matrix=True,
            seed=seed + i,
        )

    return noise_array


def plot_dataset(*, dataset):
    def get_plot_args(dataset):

        (
            X,
            true_labels_train,
            X_test,
            true_labels_test,
            labels,
            noisy_test_labels,
            label_errors_mask,
            test_label_errors_mask,
            multiple_errors_mask_dict,
            test_multiple_errors_mask_dict,
            ps,
            py,
            noise_matrix,
            m,
            n,
        ) = (dataset[key] for key in dataset.keys())
        return (
            X,
            true_labels_train,
            X_test,
            true_labels_test,
            labels,
            noisy_test_labels,
            label_errors_mask,
            test_label_errors_mask,
            multiple_errors_mask_dict,
            test_multiple_errors_mask_dict,
            ps,
            py,
            noise_matrix,
            m,
            n,
        )

    (
        X,
        true_labels_train,
        X_test,
        true_labels_test,
        labels,
        noisy_test_labels,
        label_errors_mask,
        test_label_errors_mask,
        multiple_errors_mask_dict,
        test_multiple_errors_mask_dict,
        ps,
        py,
        noise_matrix,
        m,
        n,
    ) = get_plot_args(dataset)

    # Binarized labels to unique label indices
    unique_labels = np.unique(
        np.concatenate([true_labels_train, true_labels_test, labels]), axis=0
    )
    label_to_index = {tuple(label): i for i, label in enumerate(unique_labels)}

    # Plot the features with and without label noise
    x_axis, y_axis = 0, 1

    fig, ax = plt.subplots(1, 3, figsize=(16, 10), dpi=200)
    # Get mean class-wise label error rate
    label_error_rates = true_labels_train != labels
    label_error_rates = np.sum(label_error_rates, axis=1) / true_labels_train.shape[1]
    mean_label_error_rate = np.mean(label_error_rates)

    # Plot the noisy labels
    ax[0].scatter(
        X[:, x_axis], X[:, y_axis], c=[label_to_index[tuple(label)] for label in labels]
    )
    # Annotated with the mean label error rate
    ax[0].annotate(
        f"Mean label error rate: {mean_label_error_rate:.2f}",
        xycoords="axes fraction",
        xy=(0.45, 0.95),
    )
    ax[0].set_title("Features with label noise")

    # Plot the true labels
    ax[1].scatter(
        X[:, x_axis],
        X[:, y_axis],
        c=[label_to_index[tuple(label)] for label in true_labels_train],
    )
    ax[1].set_title("Features without label noise")

    overall_label_error_rate = np.any(true_labels_train != labels, axis=1).mean()
    # Get overall label error rate
    # Plot the label error mask
    ax[2].scatter(X[:, x_axis], X[:, y_axis], c=label_errors_mask)
    # Annotated with the overall label error rate
    ax[2].annotate(
        f"Total label error rate: {overall_label_error_rate:.2f}",
        xycoords="axes fraction",
        xy=(0.5, 0.95),
    )
    ax[2].set_title("Label error mask")

    return fig, ax


def make_dataset(
    *,
    seed: int,
    dataset_kwargs: dict,
    output_dir: str,
    dataset_name: Optional[str] = None,
):
    dataset = generate_noisy_dataset(**{**dataset_kwargs, "seed": seed})
    fig, ax = plot_dataset(dataset=dataset)
    # Pad seed to 4 digits
    if dataset_name is None:
        dataset_name = f"dataset_{str(seed).zfill(4)}"
    pickle_file = pathlib.Path(output_dir) / f"{dataset_name}.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(dataset, f)

    image_name = f"{dataset_name}.png"
    image_file = IMAGE_DIR / image_name
    # Save the figure in the image directory
    # make sure the directory exists
    pathlib.Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(image_file, bbox_inches="tight", dpi=200)


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
            K = kwargs["n_classes"]
            avg_traces = 1 - np.random.gamma(shape, scale, size=K)
            weights = 1 - (np.exp(-np.arange(K) ** 2 / K) / (2 * K))
            avg_traces[np.argsort(avg_traces)] *= weights
            avg_traces = np.maximum(avg_traces, 1 - avg_traces)
            kwargs["avg_trace"] = avg_traces
            make_dataset(
                seed=seed,
                dataset_kwargs=kwargs,
                output_dir=OUTPUT_DIR,
                dataset_name=dataset_name,
            )
