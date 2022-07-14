import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D


color_map = {
    "ME": "C0",
    "MEF": "C1",
    "RO": "C2",
    "ROF": "C3",
    "LS": "C4",
}

variant_map = {
    "RAND": "-.",
    "ADAPT": ":",
    "ANTI": "--",
    "SGD": "s-",
    "NONE": "-",
}


def compute_bias(attr, truth):
    """Compute bias in attribution relative to ground truth.

    Args:
        attr: attribution array of shape (# iterations, # features)
        truth: ground truth array of shape (# features)
    """
    return np.linalg.norm(attr.mean(0) - truth[None, :]) ** 2


def compute_variance(attr):
    """Compute variance across attributions.

    Args:
        attr: attribution array of shape (# iterations, # features)
    """
    return (np.linalg.norm(attr - attr.mean(0), axis=1) ** 2).mean()


def compute_error(attr, truth):
    """Compute error in attribution relative to ground truth.

    Args:
        attr: attribution array of shape (# iterations, # features)
        truth: ground truth array of shape (# features)
    """
    return np.square(attr - truth[None, :]).sum(1).mean()


def gather_results(num_evals_lst, exact_attribution, attributions):
    """Compute bias, variance, and error of attributions.

    Args:
        num_evals_lst: list of number of evaluations per iteration
        exact_attribution: ground truth shap (# features)
        attributions: estimates shape (# iterations, # features)

    Returns:
        var_df: variances per method and num_evals
        bias_df: biases per method and num_evals
        err_df: errors per method and num_evals
    """

    # Compute and create result dataframes
    errors = {num_evals: {} for num_evals in num_evals_lst}
    biases = {num_evals: {} for num_evals in num_evals_lst}
    variances = {num_evals: {} for num_evals in num_evals_lst}

    for key, attribution in attributions.items():
        errors[key[0]][key[1]] = compute_error(attribution, exact_attribution)
        biases[key[0]][key[1]] = compute_bias(attribution, exact_attribution)
        variances[key[0]][key[1]] = compute_variance(attribution)

    var_df = pd.DataFrame.from_dict(variances)
    bias_df = pd.DataFrame.from_dict(biases)
    err_df = pd.DataFrame.from_dict(errors)

    # Convert indices to upper case
    index = [x.upper() for x in var_df.index]
    var_df.index = bias_df.index = err_df.index = index

    # Set nan where there were insufficient samples
    is_invalid = pd.isna(var_df) | (var_df == 0)
    bias_df[is_invalid] = np.nan
    err_df[is_invalid] = np.nan
    var_df[is_invalid] = np.nan

    return (var_df, bias_df, err_df)


def plot_error_variants(num_evals_lst, err_df, cache_path):
    """Plot and save all error variants."""

    base_methods = ["ME", "MEF", "ROF", "RO", "LS"]

    method_map = {
        "ME": (0, 0),
        "MEF": (1, 0),
        "RO": (0, 1),
        "ROF": (1, 1),
        "LS": (0, 2),
    }

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(8, 5.5))

    for i, base_method in enumerate(base_methods):
        for index, row in err_df.iterrows():

            curr_method = index.split("_")[0]

            variant = "NONE"
            if "_" in index:
                variant = index.split("_")[1]

            if index == base_method or curr_method == base_method:

                row_ind, col_ind = method_map[curr_method]
                ax = axs[row_ind][col_ind]

                ax.plot(
                    np.log10(num_evals_lst),
                    np.log10(row),
                    variant_map[variant],
                    label=index,
                    color=color_map[curr_method],
                )

                ax.set_xlabel("$log_{10}$(# Samples)")
                ax.set_ylabel("$log_{10}$(Error)")
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)

    #             ax.legend()

    fig.delaxes(axs[1][2])

    # Form the legends
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    def create_line(line_type="-", marker=None, color="black"):
        return Line2D(
            [0], [0], linestyle=line_type, color=color, marker=marker, lw=2
        )

    method_dict = {
        "Multilinear": create_line(color="C0"),
        "Multilinear (feature)": create_line(color="C1"),
        "Random Order": create_line(color="C2"),
        "Random Order (feature)": create_line(color="C3"),
        "Least Squares": create_line(color="C4"),
    }
    fig.legend(
        handles=method_dict.values(),
        labels=method_dict.keys(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.1),
        frameon=False,
    )

    variant_dict = {
        "Default variant": create_line(),
        "Adaptive sampling": create_line(":"),
        "Antithetic sampling": create_line("--"),
        "Stochastic gradient descent": create_line("-", "s"),
        "Random q": create_line("-."),
    }
    fig.legend(
        handles=variant_dict.values(),
        labels=variant_dict.keys(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0),
        frameon=False,
    )

    plt.savefig(cache_path + "all_error.pdf")
    plt.show()


def plot_indices(
    num_evals_lst,
    indices,
    result_df,
    ylabel,
    title,
    cache_path,
    fname,
    ylim=None,
):
    """Plot specific methods based on dataframe"""
    figure(figsize=(4, 3.2), dpi=80)

    for index, row in result_df.iterrows():

        if index in indices:

            plt.plot(np.log10(num_evals_lst), np.log10(row), label=index)

    plt.legend()
    plt.xlabel("$log_{10}$(# Samples)")
    plt.ylabel("$log_{10}$" + f"({ylabel})")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.savefig(cache_path + fname)
    plt.show()
