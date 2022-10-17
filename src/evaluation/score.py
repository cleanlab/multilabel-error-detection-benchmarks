from cleanlab.internal.multilabel_utils import ClassLabelScorer, MultilabelScorer
import numpy as np
import pandas as pd
import pathlib
import pickle
from functools import partial
from sklearn.metrics import average_precision_score, roc_auc_score 
import yaml
import json
from collections import defaultdict

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


def configure_aggregators(labels, *, temperatures: list, ks: list, alphas: list):
    """Configure the aggregators for the MultilabelScorer."""

    _, K = labels.shape

    temperatures = [0.01, 0.1, 1, 10, 100]
    ks = [1, 2, 3, 4, 5, 10, 20, 50, 100]
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
    num_examples, num_classes = labels.shape
    num_errors = np.sum(label_errors_mask)
    num_unique_labels = dataset["m"]
    pred_probs = pickle.load(open(pred_probs_file, "rb"))
    
    aggregators = configure_aggregators(labels, **aggregator_params)

    scorers = [
        MultilabelScorer(base_scorer=ClassLabelScorer.SELF_CONFIDENCE, aggregator=aggregator)
        for aggregator in aggregators
    ]

    experiments = []
    for exp_id, scorer in enumerate(scorers):
        scores = scorer(labels, pred_probs)
        inv_scores = 1 - scores
        auroc = roc_auc_score(label_errors_mask, inv_scores)
        lift_at_100 = lift_at_k(label_errors_mask, inv_scores, k=100)
        lift_at_num_errors = lift_at_k(label_errors_mask, inv_scores, k=num_errors)
        auprc = average_precision_score(label_errors_mask, inv_scores)
        ap_at_100 = average_precision_at_k(label_errors_mask, inv_scores, k=100)
        ap_at_num_errors = average_precision_at_k(label_errors_mask, inv_scores, k=num_errors)

        experiment = {
            "exp_id": exp_id,
            "dataset_name": dataset_name,
            "num_examples": num_examples,
            "num_classes": num_classes,
            "num_unique_labels": num_unique_labels,
            "num_errors": num_errors,
            "class_label_scorer": scorer.base_scorer,
            "aggregator": scorer.aggregator.func.__name__ if hasattr(scorer.aggregator, "func") else scorer.aggregator.__qualname__,
            "aggregator_kwargs": str(
                scorer.aggregator.keywords 
                if hasattr(scorer.aggregator, "keywords")
                else {}
            ),
            "auroc": auroc,
            "lift_at_100": lift_at_100,
            "lift_at_num_errors": lift_at_num_errors,
            "auprc": auprc,
            "ap_at_100": ap_at_100,
            "ap_at_num_errors": ap_at_num_errors,
            "scores": scores,
        }
        
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
    for dataset_file, pred_probs_file in dataset_pred_probs_pairs:
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
        
    # with open(SCORE_DIR / "results_agg.json", "w") as f:
    #     json.dump(results, f, indent=4)

    df_agg.to_json(SCORE_DIR / "results_agg.json", orient="split", indent=4)


if __name__ == "__main__":
    main()