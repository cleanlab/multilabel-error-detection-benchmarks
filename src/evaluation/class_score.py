from cleanlab.internal.multilabel_scorer import ClassLabelScorer, MultilabelScorer
import pandas as pd
import pathlib
import pickle
from tqdm import tqdm

DATA_DIR = pathlib.Path("data/generated")
PRED_PROBS_DIR = pathlib.Path("data/pred_probs")
SCORE_DIR = pathlib.Path("data/scores")

# Default aggregator doesn't matter as it won't be used in this script
SCORER = MultilabelScorer(base_scorer=ClassLabelScorer.SELF_CONFIDENCE)


def run_scorer(dataset_file, pred_probs_file, scorer=SCORER):

    dataset = pickle.load(open(dataset_file, "rb"))
    pred_probs_dict = pickle.load(open(pred_probs_file, "rb"))
    dataset_name = dataset_file.stem
    labels = dataset["labels"]

    experiments = []
    for model_name, pred_probs in pred_probs_dict.items():
        class_label_quality_scores = scorer.get_class_label_quality_scores(labels=labels, pred_probs=pred_probs)
        experiments.append(
            {
                "dataset": dataset_name,
                "model_name": model_name,
                "class_label_quality_scores": class_label_quality_scores,
            }
        )
    return experiments

def run_all_scorer_experiments():
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
        experiments = run_scorer(
            dataset_file,
            pred_probs_file,
        )
        all_experiments += experiments

    df = pd.DataFrame(all_experiments)
    return df

def main():
    df = run_all_scorer_experiments()
    df.to_pickle(SCORE_DIR / "class_scores.pkl")

if __name__ == "__main__":
    main()