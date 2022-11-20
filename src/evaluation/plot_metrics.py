import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCORE_DIR = pathlib.Path("data/scores")
IMAGE_DIR = pathlib.Path("data/images/scores")

# Make sure the image directory exists
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

METRIC_COLS_MAP = {
    "auroc": "AUROC",
    "lift_at_100": "Lift@100",
    "lift_at_num_errors": "Lift@num_errors",
    "auprc": "AUPRC",
    "ap_at_100": "AP@100",
    "ap_at_num_errors": "AP@num_errors",
    "spearman": "Spearman Rank Correlation",
    "ap_at_100_two": "AP@100 (two or more label errors)",
    "ap_at_num_errors_two": "AP@num_errors (two or more label errors)",
    "ap_at_100_three": "AP@100 (three or more label errors)",
    "ap_at_num_errors_three": "AP@num_errors (three or more label errors)",
    "auroc_two": "AUROC (two or more label errors)",
    "auroc_three": "AUROC (three or more label errors)",
    "auprc_two": "AUPRC (two or more label errors)",
    "auprc_three": "AUPRC (three or more label errors)",
}

COLUMN_MAP = {
    **METRIC_COLS_MAP,
    "exp_id": "Experiment ID",
    "aggregator": "Aggregation method",
    "aggregator_kwargs": "Aggregation method parameters",
    "dataset_name": "Dataset",
    "model_name": "Model",
    "num_errors": "Number of label errors",
    "num_examples": "Number of examples",
    "num_classes": "Number of classes",
    "num_unique_labels": "Number of unique labels",
    "class_label_scorer": "Class label scoring method",
}


def filter_by_model(df, model_name: str):
    return df[df["Model"] == model_name]

def find_best_group_kwargs_by_metric(df, metric: str, group_by: list[str], ):

    kwarg_col = "Aggregation method parameters"
    df_out = df.groupby(group_by+ [kwarg_col])[[metric]].mean()
    df_out = df_out.groupby(group_by).idxmax(axis=0)
    df_out = df_out.apply(lambda x: x[0], axis=1)
    return df_out

def get_plot_df(
    df: pd.DataFrame,
    model: str,
    metric: str,
    group_by: list[str],
):
    df_model = filter_by_model(df, model)
    df_model_best_kwargs = find_best_group_kwargs_by_metric(df_model, metric, group_by)
    groups = df.groupby(group_by + ["Aggregation method parameters"]).groups
    indices = pd.Index(np.concatenate([groups[group] for group in df_model_best_kwargs]))
    plot_df = df_model[df_model.index.isin(indices)]
    return plot_df

def get_ema_plot_df(
    df: pd.DataFrame,
    model: str,
):
    df_model = filter_by_model(df, model)
    plot_df = df_model[df_model["Aggregation method"] == "exponential_moving_average"]
    return plot_df

def format_image_filename(model: str, metric: str):
    model_str = model.lower().replace(" ", "_")
    metric_str = metric.lower().replace(" ", "_")
    return f"{model_str}_{metric_str}.png"

def generate_metric_plots(
    df: pd.DataFrame,
    model: str,
    metrics: list[str],
    group_by: list[str],
    hue: str = "Dataset size",
    prefix: str = "",
):
    for metric in metrics:
        plt.figure(figsize=(24, 18), dpi=400)
        ax = plot_swarm(df, model, metric, group_by, hue)

        plt.savefig(IMAGE_DIR / (prefix + format_image_filename(model, metric)))
        plt.close()

def plot_swarm(
    df: pd.DataFrame,
    model: str,
    metric: str,
    group_by: list[str],
    hue: str = "Dataset size",
):
    x = group_by[-1]
    ax = sns.swarmplot(data=df, x=x, y=metric, hue=hue, dodge=True, size=9)
    ax.set_title(f"{model} - {metric} by '{x}' and '{hue}'")
    plt.setp(ax.get_xticklabels(), rotation=20, fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    # Set font size of title, axis labels and legend
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    ax.legend(fontsize=18)
    
    ax.set_ylabel(metric)
    ax.set_xlabel(group_by[0])

    for i in range(df[x].nunique()):
        ax.axvline(i + 0.5, color="gray", c=".5", linestyle="--", linewidth=1)
    return ax

def main():
    df = pd.read_csv(SCORE_DIR / "results.csv")

    df["Dataset size"] = df["dataset_name"].apply(lambda x: x.split("_")[0])

    # Drop all columns containing the substring "lift_"
    df.drop(columns=[col for col in df.columns if "lift_" in col], inplace=True)

    # Rename columns
    df.rename(columns=COLUMN_MAP, inplace=True)

    # Rename values maps
    size_map = {
        "small": "Small",
        "large": "Large",
    }

    model_map = {
        "log_reg": "Logistic Regression",
        "rf": "Random Forest",
    }

    # Rename values
    for col, map_dict in zip(["Dataset size", "Model"], [size_map, model_map]):
        df[col] = df[col].map(map_dict)


    # Types of models to plot
    models = ["Logistic Regression", "Random Forest"]

    # Metric to chose best set of hyperparameters per aggregation method for plotting
    plot_df_metric = "AP@num_errors"
    
    # Types of metrics to plot
    metrics = [
        "AUROC",
        "AUPRC",
        "AP@100",
        "AP@num_errors",
        "AP@num_errors (two or more label errors)",
        "AP@num_errors (three or more label errors)",
        "Spearman Rank Correlation",
    ]

    # Types of groupings to plot
    group_by = ["Dataset size", "Aggregation method"]

    sns.set_theme(style="whitegrid", palette="muted")
    for model in models:
        plot_df = get_plot_df(df, model, plot_df_metric, group_by)
        generate_metric_plots(plot_df, model, metrics, group_by)


    # Create other swarm plots for exponential moving average with different hues for hyperparameters
    group_by = ["Dataset size"]
    hue = "Aggregation method parameters"
    metrics = [
        "AP@num_errors",
        "Spearman Rank Correlation",
    ]
    for model in models:
        plot_df = get_ema_plot_df(df, model)
        generate_metric_plots(plot_df, model, metrics, group_by, hue=hue, prefix="ema_")

if __name__ == "__main__":
    main()
