import ast
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
    "ap_at_num_errors": "AP@T",
    "spearman": "Spearman Rank Correlation",
    "ap_at_100_two": "AP@100 (two or more label errors)",
    "ap_at_num_errors_two": "AP@T (two or more label errors)",
    "ap_at_100_three": "AP@100 (three or more label errors)",
    "ap_at_num_errors_three": "AP@T (three or more label errors)",
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

def drop_experiments_for_single_dataset_size(
    df: pd.DataFrame,
    groups: list[str] = ["Dataset size"],
    col: str = "Aggregation method parameters",
) -> pd.DataFrame:
    # Find unique `col` for each group
    unique_kwargs = df.groupby(groups)[col].unique().apply(set)
    
    # Intersection for `col` that are used for all groups
    common_kwargs = set.intersection(*unique_kwargs)
    
    # Select rows with col that are in the intersection  
    return df[df[col].apply(lambda x: x in common_kwargs)]

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
    manual_kwargs: dict[str, str] = None,
):
    df_model = filter_by_model(df, model)
    df_model_best_kwargs = find_best_group_kwargs_by_metric(df_model, metric, group_by)

    if manual_kwargs is not None:
        # Use keys from `manual_kwargs` to override the best kwargs
        df_model_best_kwargs = df_model_best_kwargs.apply(
            lambda x: manual_kwargs.get(x[0], x)
        )

    groups = df.groupby(group_by + ["Aggregation method parameters"]).groups
    indices = pd.Index(np.concatenate([groups[group] for group in df_model_best_kwargs]))
    plot_df = df_model[df_model.index.isin(indices)]


    # Save the plotted kwargs in a csv file
    plots_kwargs_df = df_model_best_kwargs.to_frame()
    plots_kwargs_df.columns = ["Plot kwargs"]
    plots_kwargs_df["Plot kwargs"] = plots_kwargs_df["Plot kwargs"].apply(
        lambda x: x[-1]
    )

    plots_kwargs_df.to_csv(IMAGE_DIR / f"{model.lower().replace(' ', '_')}_best_kwargs.csv")

    return plot_df

def get_ema_plot_df(
    df: pd.DataFrame,
    model: str,
    ema_metric: str = "EMA"
):
    df_model = filter_by_model(df, model)
    plot_df = df_model[df_model["Aggregation method"] == ema_metric]
    return plot_df

def format_image_filename(model: str, metric: str):
    model_str = model.lower().replace(" ", "_")
    metric_str = metric.lower().replace(" ", "_")
    return f"{model_str}_{metric_str}.pdf"

def generate_metric_plots(
    df: pd.DataFrame,
    model: str,
    metrics: list[str],
    group_by: list[str],
    hue: str = "Dataset size",
    prefix: str = "",
    legend_title: str = "",
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 100,
    rotation: int = 30,
):
    for metric in metrics:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plot_swarm(df, metric, group_by, hue, rotation=rotation)
        if legend_title:
            ax.legend(title=legend_title)

        plt.savefig(IMAGE_DIR / (prefix + format_image_filename(model, metric)), bbox_inches='tight', pad_inches=2)
        plt.close()

def plot_swarm(
    df: pd.DataFrame,
    metric: str,
    group_by: list[str],
    hue: str = "Dataset size",
    fs: int = 16,
    rotation: int = 30,
):
    x = group_by[-1]
    ax = sns.swarmplot(data=df, x=x, y=metric, hue=hue, dodge=True, size=5, legend="full")
    plt.setp(ax.get_xticklabels(), rotation=rotation, fontsize=fs)
    plt.setp(ax.get_yticklabels(), fontsize=fs)

    # Set font size of title, axis labels and legend
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.legend(fontsize=fs)
    
    ax.set_xlabel("")
    ax.set_ylabel(metric)

    for i in range(df[x].nunique()):
        ax.axvline(i + 0.5, color="gray", c=".5", linestyle="--", linewidth=1)
    return ax

def load_data():
    df = pd.read_csv(SCORE_DIR / "results.csv")

    df["Dataset size"] = df["dataset_name"].apply(lambda x: x.split("_")[0])

    # Drop all columns containing the substring "lift_"
    df.drop(columns=[col for col in df.columns if "lift_" in col], inplace=True)

    # Rename columns
    df.rename(columns=COLUMN_MAP, inplace=True)
    return df


def preprocess_data():
    df = load_data()

    # Rename values maps
    size_map = {
        "small": "Small Dataset",
        "large": "Large Dataset",
    }

    model_map = {
        "log_reg": "Logistic Regression",
        "rf": "Random Forest",
    }

    aggregator_map = {
        "amin": "Min",
        "amax": "Max",
        "mean": "Mean",
        "median": "Median",
        "softmin_pooling": "Softmin",
        "log_transform_pooling": "Log",
        "exponential_moving_average": "EMA",
        "cumulative_average": "Cumulative",
        "simple_moving_average": "SMA",
        "weighted_cumulative_average": "Weighted",
    }



    ema_aggregator_kwargs_map = {
        kwargs: ast.literal_eval(kwargs)["alpha"]
        for kwargs in (
            df[
                df["Aggregation method"]
                .str
                .contains("exponential_moving_average")
            ]["Aggregation method parameters"]
        )  
    }
    ema_aggregator_kwargs_map.update({
        "{'alpha': None}": "2/(K+1)",
        "{'weights': None}": "weights: exponential decay",
    })

    # Rename values
    for col, map_dict in zip(
        ["Dataset size", "Model", "Aggregation method", "Aggregation method parameters"],
        [size_map, model_map, aggregator_map, ema_aggregator_kwargs_map],
    ):
        df[col] = df[col].map(map_dict).fillna(df[col])

    df = drop_experiments_for_single_dataset_size(df)
    return df

def main():
    df = preprocess_data()

    # Types of models to plot
    models = ["Logistic Regression", "Random Forest"]

    # Metric to chose best set of hyperparameters per aggregation method for plotting
    plot_df_metric = "AP@T"
    
    # Types of metrics to plot
    metrics = [
        "AUPRC",
        "AP@T",
        "AP@T (two or more label errors)",
        "AP@T (three or more label errors)",
        "Spearman Rank Correlation",
    ]

    # Types of groupings to plot
    group_by = ["Dataset size", "Aggregation method"]

    sns.set_theme(style="whitegrid", palette="muted")
    manual_kwargs = {
        "EMA": ("EMA", 0.8),
        "cumulative_average": ("cumulative_average", "{'k': 2}"),
        "simple_moving_average": ("simple_moving_average", "{'k': 2}"),
    }
    for model in models:
        plot_df = get_plot_df(df, model, plot_df_metric, group_by[1:], manual_kwargs=manual_kwargs)
        generate_metric_plots(plot_df, model, metrics, group_by)


    # Create other swarm plots for exponential moving average with different hues for hyperparameters
    group_by = ["Dataset size"]
    hue = "Aggregation method parameters"
    metrics = [
        "AP@T",
        "AP@T (two or more label errors)",
        "AP@T (three or more label errors)",
        "Spearman Rank Correlation",
    ]
    for model in models:
        plot_df = get_ema_plot_df(df, model)
        generate_metric_plots(plot_df, model, metrics, group_by, hue=hue, prefix="ema_", legend_title="alpha", rotation=0)

if __name__ == "__main__":
    main()
