import pandas as pd
import pathlib


OUTPUT_DIR = pathlib.Path("data/accuracy")
def get_group_statistics(df):
    stats_columns = ["Accuracy", "Average accuracy", "Jaccard score"]
    df_group = df.groupby(["dataset_group", "clf", "train_set", "test_set"])[stats_columns].agg(["mean", "std"])
    for column in stats_columns:
        df_group[column, "std"] = df_group[column, "std"].apply(lambda x: "{:.0e}".format(x)).apply(lambda x: float(x))
        # Keep same number of digits for the mean based on the first significant digit of the std
        df_group[column, "mean"] = df_group[column, "mean"].apply(lambda x: "{:.{}f}".format(x, len(str(df_group[column, "std"][0]).split(".")[1]))).apply(lambda x: float(x))

    return df_group


if __name__ == "__main__":
    df = pd.read_csv(OUTPUT_DIR / "results.csv")
    df_group_stats = get_group_statistics(df)
    df_group_stats.columns = [" ".join(col).strip() for col in df_group_stats.columns.values]
    df_group_stats.to_csv(OUTPUT_DIR / "results_group.csv")
