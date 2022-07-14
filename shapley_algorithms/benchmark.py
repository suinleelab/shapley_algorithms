import os
import shap
import time
import pickle
import xgboost
import numpy as np
import pandas as pd

from shapley_algorithms.explain import Exact
from shapley_algorithms.explain import MultilinearFeature
from shapley_algorithms.explain import Multilinear
from shapley_algorithms.explain import RandomOrderFeature
from shapley_algorithms.explain import RandomOrder
from shapley_algorithms.explain import LeastSquares
from shapley_algorithms.explain import LeastSquaresSGD
from functools import partial


def cache_train(train_fn, cache_path, X, y):
    """Cached version of training a model.

    Args:
      train_fn: function to train and return a model based on X and y.
      cache_path: path the cache the model.
      X: training inputs.
      y: training labels.

    Returns:
      model: trained model (may be loaded from cache).
    """
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    model_name = cache_path + "model.pkl"
    if not os.path.exists(model_name):
        model = train_fn(X, y)
        pickle.dump(model, open(model_name, "wb"))
    else:
        model = pickle.load(open(model_name, "rb"))

    return model


def compute_attributions(
    model,
    num_evals_lst,
    explicand,
    baselines,
    X,
    y,
    cache_path,
    num_iter=100,
    output_type="regression",
    skip_methods=None,
):
    """Compute attributions for many iterations

    Args:
      model: xgboost model being explained
      num_evals_lst: number of evaluations for each stochastic estimator
      explicand: sample being explained, shape (num_features)
      baselines: sample being compared to, shape (1, num_features)
      X: sample inputs
      y: sample labels
      cache_path: path to cache attributions and statistics to
      num_iter: number of iterations to compute bias/variance over
      output_type: regression or classification model
      skip_methods: methods to skip in benchmarking

    Returns:
      exact_attribution: exact baseline shapley for explicand and baseline
      attributions: dictionary of attributions, where keys are tuples of
        number of evals and method names
      evals_counts: dictionary of actual number of evaluations per approach
      runtimes: dictionary of runtimes of each approach
    """

    # Wrap the model prediction to accept numpy arrays
    if output_type == "regression":

        def model_wrapper(model, x):
            """Wrap regression model"""
            if len(x.shape) == 1:
                x = x[None, :]
            return model.predict(pd.DataFrame(x, columns=X.columns))

    elif output_type == "classification":

        def model_wrapper(model, x):
            """Wrap classification model to output log odds"""
            if len(x.shape) == 1:
                x = x[None, :]
            x2 = pd.DataFrame(x, columns=X.columns)
            preds = model.predict(xgboost.DMatrix(x2))
            lodds = np.log(preds / (1 - preds))
            return lodds

    else:

        raise ValueError(
            'output_type must be ["regression", "classification"], '
            f"not {output_type}"
        )

    num_features = baselines.shape[1]
    model_fn = partial(model_wrapper, model)

    # Form algorithms to test
    algorithms = {
        "me": partial(Multilinear(model_fn, num_features)),
        "me_rand": partial(
            Multilinear(model_fn, num_features), samples_per_prob=None
        ),
        "me_anti": partial(
            Multilinear(model_fn, num_features), is_antithetic=True
        ),
        "mef": partial(MultilinearFeature(model_fn, num_features)),
        "mef_rand": partial(
            MultilinearFeature(model_fn, num_features), samples_per_prob=None
        ),
        "mef_adapt": partial(
            MultilinearFeature(model_fn, num_features), is_adaptive=True
        ),
        "mef_anti": partial(
            MultilinearFeature(model_fn, num_features), is_antithetic=True
        ),
        "rof": partial(RandomOrderFeature(model_fn, num_features)),
        "rof_adapt": partial(
            RandomOrderFeature(model_fn, num_features), is_adaptive=True
        ),
        "rof_anti": partial(
            RandomOrderFeature(model_fn, num_features), is_antithetic=True
        ),
        "ro": RandomOrder(model_fn, num_features),
        "ro_anti": partial(
            RandomOrder(model_fn, num_features), is_antithetic=True
        ),
        "ls": LeastSquares(model_fn, num_features),
        "ls_anti": partial(
            LeastSquares(model_fn, num_features), is_antithetic=True
        ),
        "ls_sgd": partial(LeastSquaresSGD(model_fn, num_features, y.max())),
    }

    # Exact approach (assuming a tree model)
    if num_features < 16:
        algorithms["exact"] = Exact(model_fn, num_features)
    else:

        def tree_shap(explicand, baselines):
            if output_type == "regression":
                explainer = shap.TreeExplainer(model, baselines)
            elif output_type == "classification":
                explainer = shap.TreeExplainer(model, baselines)
            return explainer(explicand[None, :]).values

        algorithms["exact"] = tree_shap

    exact_attribution = algorithms["exact"](explicand, baselines)

    # Benchmark for many numbers of evaluations
    attributions = {}
    evals_counts = {}
    runtimes = {}

    for method, algorithm in algorithms.items():

        # Skip exact approach
        if method == "exact":
            continue

        # Go through all stochastic estimators
        for num_evals in num_evals_lst:

            cache_name = f"{cache_path}/{num_evals}_{method}.p"

            if os.path.exists(cache_name):

                print(f"Loading from cache: {(num_evals, method)}")

                (
                    attributions[(num_evals, method)],
                    evals_counts[(num_evals, method)],
                    runtimes[(num_evals, method)],
                ) = pickle.load(open(cache_name, "rb"))

            else:

                if skip_methods and method in skip_methods:

                    print(
                        "Skipping running from scratch: "
                        f"{(num_evals, method)}"
                    )
                    continue

                print(f"Running from scratch: {(num_evals, method)}")

                attributions[(num_evals, method)] = np.zeros(
                    (num_iter, X.shape[1])
                )
                evals_counts[(num_evals, method)] = np.zeros((num_iter))
                runtimes[(num_evals, method)] = np.zeros((num_iter))

                for i in range(num_iter):

                    start_time = time.time()
                    attribution, evals_count = algorithm(
                        explicand, baselines, num_evals=num_evals
                    )
                    elapsed_time = time.time() - start_time

                    # Keep track of statistics
                    attributions[(num_evals, method)][i] = attribution
                    evals_counts[(num_evals, method)][i] = evals_count
                    runtimes[(num_evals, method)][i] = elapsed_time

                pickle.dump(
                    (
                        attributions[(num_evals, method)],
                        evals_counts[(num_evals, method)],
                        runtimes[(num_evals, method)],
                    ),
                    open(cache_name, "wb"),
                )

    return exact_attribution, attributions, evals_counts, runtimes
