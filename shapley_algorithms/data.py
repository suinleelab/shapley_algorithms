from sklearn import datasets
import pandas as pd
import numpy as np
import shap
import os


def diabetes():
    """Load sklearn diabetes toy data.  Small regression dataset.
    """

    dataset = datasets.load_diabetes()
    X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
    y = dataset["target"]

    return (X, y)


def nhanes():
    """Load NHANES mortality data.  Medium classification dataset.

    From: https://www.cdc.gov/nchs/nhanes/index.htm
    Processed by: https://github.com/slundberg/shap.
    """

    fname_prefix = "../../data/"
    X = pd.read_csv(fname_prefix + "NHANESI_X.csv", index_col=0)
    y = pd.read_csv(fname_prefix + "NHANESI_y.csv", index_col=0)["y"].values

    # Filter out unknown labels for 5-year mortality
    is_unknown = (y < 0) & (y > -5)
    y = y[~is_unknown]

    # Convert to 5-year mortality classification problem
    y_binary = (y >= 0) & (y < 5)

    # Convert bool columns to int due to issues with SHAP tree ensemble
    for j, dtype in enumerate(X.dtypes):
        if dtype == "bool":
            X.iloc[:, j] = X.iloc[:, j].replace({True: 1, False: 0})

    return (X, y_binary)


def blog():
    """Load UCI blog feedback data.  Large regression dataset.

    https://archive.ics.uci.edu/ml/datasets/BlogFeedback
    """
    data_url = "https://archive.ics.uci.edu/ml/datasets/BlogFeedback"
    fname = "../../data/blogData_train.csv"

    if not os.path.exists(fname):
        raise RuntimeError(
            f"Please download data from {data_url} "
            "and put it in shapley_algorithms/data/"
        )

    data = pd.read_csv(fname, header=None)
    X = data.iloc[:, :280]
    y = data.iloc[:, 280:]

    return (X, y)
