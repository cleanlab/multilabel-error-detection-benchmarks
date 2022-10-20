from typing import Optional
from cleanlab.internal.multilabel_utils import ClassLabelScorer, MultilabelScorer
import numpy as np
import pandas as pd
import pathlib
import pickle
from functools import partial
from sklearn.metrics import average_precision_score, roc_auc_score 
import yaml
from collections import defaultdict
from tqdm import tqdm

from src.evaluation.aggregate import (
    softmin_pooling,
    log_transform_pooling,
    cumulative_average,
    simple_moving_average,
    exponential_moving_average,
    weighted_cumulative_average,
)
from src.evaluation.metrics import average_precision_at_k, lift_at_k 

DATA_DIR = pathlib.Path("data/generated")
PRED_PROBS_DIR = pathlib.Path("data/pred_probs")
SCORE_DIR = pathlib.Path("data/scores")


def configure_aggregators(
    labels,
    *,
    temperatures: Optional[list] = None,
    ks: Optional[list] = None,
    alphas: Optional[list] = None,
):
    """Configure the aggregators for the MultilabelScorer."""

    _, K = labels.shape

    if temperatures is None:
        temperatures = [0.01, 0.1, 1, 10, 100]
    if ks is None:
        ks = [1, 2, 3, 4, 5, 10, 20, 50, 100]
    if alphas is None:
        alphas = [0.2, 0.4, 0.5, 0.6, 0.8]

    numpy_stats = [np.mean, np.median, np.max, np.min]

    softmin_temperatures = [
        partial(softmin_pooling, temperature=temperature)
        for temperature in temperatures
    ]

    log_pools = [
        partial(log_transform_pooling, weights=weights, biases=biases)
        for weights in [1]
        for biases in [0]
    ]

    cumulative_average_ks = [
        partial(cumulative_average, k=k) for k in ks
        if k <= K
    ]

    simple_moving_average_ks = [
        partial(simple_moving_average, k=k) for k in ks
        if k <= K
    ]

    exponential_moving_average_alphas = [
        partial(exponential_moving_average, alpha=alpha)
        for alpha in [None, *alphas]
        if alpha is not None or 2/(K + 1) not in alphas
    ]

    weighted_cumulative_average_weights = [
        partial(weighted_cumulative_average, weights=weights)
        for weights in [None, 1/K]
    ]

    aggregators = [
        *numpy_stats,
        *softmin_temperatures,
        *log_pools,
        *cumulative_average_ks,
        *simple_moving_average_ks,
        *exponential_moving_average_alphas,
        *weighted_cumulative_average_weights,
    ]

    return aggregators

def run_experiments(dataset_file, pred_probs_file, *, aggregator_params: dict):
    """Configure MultilabelScorer with."""

    dataset = pickle.load(open(dataset_file, "rb"))
    dataset_name = dataset_file.name
    labels = dataset["labels"]
    label_errors_mask = dataset["label_errors_mask"]
    multiple_errors_mask_dict = dataset["multiple_errors_mask_dict"]
    two_label_errors_mask = multiple_errors_mask_dict[1]
    three_label_errors_mask = multiple_errors_mask_dict[2]
    
    num_examples, num_classes = labels.shape
    num_errors = np.sum(label_errors_mask)
    num_two_label_errors = np.sum(two_label_errors_mask)
    num_three_label_errors = np.sum(three_label_errors_mask)
    num_unique_labels = dataset["m"]
    pred_probs = pickle.load(open(pred_probs_file, "rb"))
    
    aggregators = configure_aggregators(labels, **aggregator_params)

    scorers = [
        MultilabelScorer(base_scorer=ClassLabelScorer.SELF_CONFIDENCE, aggregator=aggregator)
        for aggregator in aggregators
    ]

    experiments = []
    for exp_id, scorer in tqdm(enumerate(scorers)):
        scores = scorer(labels, pred_probs)
        inv_scores = 1 - scores

        experiment = {
            "exp_id": exp_id,
            "dataset_name": dataset_name,
            "num_examples": num_examples,
            "num_classes": num_classes,
            "num_unique_labels": num_unique_labels,
            "num_errors": num_errors,
            "num_two_label_errors": np.sum(two_label_errors_mask),
            "num_three_label_errors": np.sum(three_label_errors_mask),
            "class_label_scorer": scorer.base_scorer,
            "aggregator": scorer.aggregator.func.__name__ if hasattr(scorer.aggregator, "func") else scorer.aggregator.__qualname__,
            "aggregator_kwargs": str(
                scorer.aggregator.keywords 
                if hasattr(scorer.aggregator, "keywords")
                else {}
            ),
        }
        for mask_name_suffix, mask, k in zip(
            ["", "_two", "_three"],
            [label_errors_mask, two_label_errors_mask, three_label_errors_mask],
            [num_errors, num_two_label_errors, num_three_label_errors],

        ):


            auroc = roc_auc_score(mask, inv_scores)
            lift_at_100 = lift_at_k(mask, inv_scores, k=100)
            lift_at_num_errors = lift_at_k(mask, inv_scores, k=k)
            auprc = average_precision_score(mask, inv_scores)
            ap_at_100 = average_precision_at_k(mask, inv_scores, k=100)
            ap_at_num_errors = average_precision_at_k(mask, inv_scores, k=k)

            experiment.update({
                f"auroc{mask_name_suffix}": auroc,
                f"lift_at_100{mask_name_suffix}": lift_at_100,
                f"lift_at_num_errors{mask_name_suffix}": lift_at_num_errors,
                f"auprc{mask_name_suffix}": auprc,
                f"ap_at_100{mask_name_suffix}": ap_at_100,
                f"ap_at_num_errors{mask_name_suffix}": ap_at_num_errors,
            })

        experiment.update({
            "scores": scores,
        })        
        experiments.append(experiment)
    return experiments

def run_all_experiments(*, aggregator_params: dict):

    dataset_files = list(DATA_DIR.glob("*.pkl"))
    pred_probs_files = list(PRED_PROBS_DIR.glob("*.pkl"))

    # Pair up the datasets and pred_probs
    dataset_pred_probs_pairs = []
    for dataset_file in dataset_files:
        dataset_name = dataset_file.stem
        for pred_probs_file in pred_probs_files:
            if pred_probs_file.stem == dataset_name:
                dataset_pred_probs_pairs.append((dataset_file, pred_probs_file))

    dataset_pred_probs_pairs.sort(key=lambda x: x[0].stem)

    all_experiments = []
    for dataset_file, pred_probs_file in tqdm(dataset_pred_probs_pairs):
        experiments = run_experiments(
            dataset_file,
            pred_probs_file,
            aggregator_params=aggregator_params,
        )
        all_experiments += experiments


    df = pd.DataFrame(all_experiments)
    return df


def main():
    # Load params.yaml
    all_params = yaml.load(open("params.yaml"), Loader=yaml.FullLoader)

    aggregator_params = all_params["eval"]

    df = run_all_experiments(aggregator_params=aggregator_params)

    # Save the results to the score directory
    # Ensure the directory exists
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORE_DIR / "results.csv", index=False)

    # Group by the aggregator/kwargs
    df_grouped = df.groupby(["aggregator", "aggregator_kwargs"])

    # Aggregate the mean and std of the metrics 
    metrics_cols = ["auroc",  "lift_at_100",  "lift_at_num_errors", "auprc",  "ap_at_100",  "ap_at_num_errors"]
    df_agg = df_grouped[metrics_cols].agg(["mean"])

    # Save the aggregated results to the score directory
    results = defaultdict(lambda: defaultdict(dict))

    for idx, *values in df_agg.itertuples():
        for v in values:
            for i, k in enumerate(idx):
                if i == 0:
                    nested = results[k]
                elif i == len(idx) - 1:
                    nested[k] = v
                else:
                    nested = nested[k]
    df_agg.index = df_agg.index.map(lambda x: x[0] + x[1])

    df_agg.to_json(SCORE_DIR / "results_agg.json", indent=4)


if __name__ == "__main__":
    main()