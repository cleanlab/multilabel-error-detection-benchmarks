import pandas as pd
import numpy as np
import pathlib


OUTPUT_DIR = pathlib.Path("data/accuracy")
def get_group_statistics(df):
    stats_columns = ["Average accuracy", "Jaccard score"]
    df_group = df.groupby(["dataset_group", "clf", "train_set", "test_set"])[stats_columns].agg(["mean", "std"])
    for column in stats_columns:
        df_group[column, "std"] = df_group[column, "std"].apply(lambda x: "{:.0e}".format(x)).apply(lambda x: float(x))
        # Keep same number of digits for the mean based on the first significant digit of the std
        df_group[column, "mean"] = df_group[column, "mean"].apply(lambda x: "{:.{}f}".format(x, len(str(df_group[column, "std"][0]).split(".")[1]))).apply(lambda x: float(x))

    return df_group

def get_formatted_group_statistics(df) -> pd.DataFrame:

    def lsd(x, sd:int=1) -> int:
        """Returns the number of decimal places to the least significant digit

        Negative numbers mean that the least significant digit is to the left of the decimal point.

        This function is useful for formatting floats to the least significant digit.

        Examples
        --------
        >>> lsd(0.234)
        1
        >>> round(0.234, lsd(0.234))
        0.2
        >>> round(0.1234, lsd(0.1234))
        0.01
        >>> round(0.001634, lsd(0.001634))
        0.002
        """
        
        xl, xr = str(x).split(".")
        if xl == "0":
            # Count the number of zeros after the decimal point
            return len(xr) - len(xr.lstrip("0")) + sd
        # Count the number of digits to the left of the decimal point
        return -len(xl) + sd

    def mean_std_format(x) -> str:
        mean = np.mean(x)
        std = np.std(x)
        mean_str = str(round(mean, lsd(std))).ljust(lsd(std) + 2, "0")
        std_str = str(round(std, lsd(std)))[-1]
        formatted_string = f"{mean_str}({std_str})"
        return formatted_string
    
    group_columns = ["dataset_group", "clf", "train_set", "test_set"]
    stats_columns = ["Average accuracy", "Jaccard score"]

    df_group = df.groupby(group_columns)[stats_columns].agg(mean_std_format)
    return df_group


if __name__ == "__main__":
    df = pd.read_csv(OUTPUT_DIR / "results.csv")
    df_group_stats = get_group_statistics(df)
    df_group_stats.columns = [" ".join(col).strip() for col in df_group_stats.columns.values]
    df_group_stats.to_csv(OUTPUT_DIR / "results_group.csv")

    df_group_stats_formatted = get_formatted_group_statistics(df)
    df_group_stats_formatted.sort_index(ascending=[False, True, True, True], inplace=True)
    df_group_stats_formatted.rename(
        index={
            "small": "Small",
            "large": "Large",
            "Noisy train": "Noisy",
            "Noisy test": "Noisy",
            "True train": "True",
            "True test": "True",
        },
        inplace=True
    )
    df_group_stats_formatted.index.set_names(["Datasets", "Classifier", "Train labels", "Test labels"], inplace=True)    
    df_group_stats_formatted.to_latex(
        OUTPUT_DIR / "results_group.tex",
        column_format="llllcc",
        multirow=True,
    )
