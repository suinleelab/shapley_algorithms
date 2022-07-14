import scipy
import numpy as np
import pandas as pd

from itertools import chain, combinations
from shapkit.sgd_shapley import SGDshapley


class Estimator:
    """Class for any estimator."""

    def __init__(self, model, num_features):
        """Initialize with model."""

        self.model = model
        self.num_features = num_features

    def _explain(self, explicand, num_evals):

        raise NotImplementedError()

    def __call__(self, explicand, num_evals=100):

        raise NotImplementedError()


class FeatureEstimator(Estimator):
    """Class for estimators that explain a single feature at a time."""

    def __init__(self, model, num_features):
        """Initialize with model."""

        self.is_antithetic = False
        self.samples_per_prob = None
        super().__init__(model, num_features)

    def _sample_subset(self):
        """Return subsets for numpy indexing.

        Returns:
         subset: numpy array representing a subset (either boolean or indices).
        """

        raise NotImplementedError("Must implement _sample_subset()")

    def _subset_on_off(self, subset, feature):
        """Return subsets for numpy indexing.

        Args:
         subset: numpy array representing a subset (either boolean or indices).
         feature: feature being explained.

        Returns:
         subset_on: subset of indices representing combination with feature.
         subset_off: subset of indices representing combination w/o feature.
        """

        raise NotImplementedError("Must implement _subset_on_off()")

    def _single_feature(self, feature, explicand, baselines, num_evals):
        """Sample subsets to compute phi mean and variance estimates.

        Args:
         feature: index of feature being explained.
         explicand: sample being explained.
         baselines: samples to compare to.
         num_evals: number of evaluations to aim for.
         is_antithetic: whether or not to use paired sampling.

        Returns:
         phi_mean: estimates of shapley value feature attributions.
         phi_var: variances of the marginal contributions.
         evals_count: number of model evaluations performed.
        """
        if num_evals <= 1:
            return (0, 0, 0)

        # Round to the nearest multiple of 2 or 4
        if self.is_antithetic:
            num_evals = int(4 * (num_evals // 4))
        else:
            num_evals = int(2 * (num_evals // 2))

        # Keep stack of probabilities to draw from for Multilinear
        if self.samples_per_prob:

            adj_num_evals = num_evals // (self.samples_per_prob * 2)

            if self.is_antithetic:
                adj_num_evals = num_evals // (self.samples_per_prob * 4)

            if adj_num_evals > 1:

                self.probs = np.arange(0, adj_num_evals) / (adj_num_evals - 1)
                self.probs = list(np.repeat(self.probs, self.samples_per_prob))

        # Create masked samples for on and off evaluations
        baseline_inds = np.random.randint(0, len(baselines), num_evals)
        masked_samples = baselines[baseline_inds]

        if self.is_antithetic:

            for i in range(0, num_evals // 2, 2):

                subset = self._sample_subset()
                subset_on, subset_off = self._subset_on_off(subset, feature)
                masked_samples[i, subset_on] = explicand[subset_on]
                masked_samples[num_evals // 2 + i, subset_off] = explicand[
                    subset_off
                ]

                subset = self._invert_subset(subset)
                subset_on, subset_off = self._subset_on_off(subset, feature)
                masked_samples[i + 1, subset_on] = explicand[subset_on]
                masked_samples[num_evals // 2 + i + 1, subset_off] = explicand[
                    subset_off
                ]

        else:

            for i in range(0, num_evals // 2):

                subset = self._sample_subset()
                subset_on, subset_off = self._subset_on_off(subset, feature)
                masked_samples[i, subset_on] = explicand[subset_on]
                masked_samples[num_evals // 2 + i, subset_off] = explicand[
                    subset_off
                ]

        # Compute marginal contributions
        preds = self.model(masked_samples)
        preds_on = preds[: num_evals // 2]
        preds_off = preds[num_evals // 2 :]
        deltas = preds_on - preds_off

        return np.mean(deltas, 0), np.var(deltas, 0), len(preds)

    def _explain(self, explicand, baselines, num_evals):
        """Estimate shapley value explanations by sampling combinations.

        https://arxiv.org/abs/2010.12082

        Args:
         explicand: sample being explained.
         baselines: samples being compared to.
         num_evals: number of subsets to evaluate for.

        Returns:
         phi: estimates of shapley value feature attributions.
         evals_count: total number of model evaluations.
        """

        evals_count = 0

        # Adaptive sampling using two rounds to reduce variance
        if self.is_adaptive:

            phi = np.zeros(explicand.shape)
            var = np.zeros(explicand.shape)

            # Round one uses an equal split of half of total evals
            num_evals_each1 = (num_evals // 2) // self.num_features

            for i in range(self.num_features):
                phi[i], var[i], evals_count_curr = self._single_feature(
                    i, explicand, baselines, num_evals_each1
                )
                evals_count += evals_count_curr

            # Allocate samples according to the variance in round one
            num_evals_left = num_evals - evals_count
            num_evals_each2 = num_evals_left * (var / var.sum())
            num_evals_each2 = num_evals_each2.astype("int")

            # Divy up any remaining num_evals
            for i in range(num_evals_left - sum(num_evals_each2)):
                num_evals_each2[i % self.num_features] += 1

            # Round two allocates optimally
            for i in range(self.num_features):

                if num_evals_each2[i] != 0:
                    phi_i, var_i, evals_count_i = self._single_feature(
                        i, explicand, baselines, num_evals_each2[i]
                    )

                    def reweight(v1, v2, w1, w2):
                        # Reweight values according to weights
                        return (v1 * w1 + v2 * w2) / (w1 + w2)

                    phi[i] = reweight(
                        phi[i], phi_i, num_evals_each1, num_evals_each2[i]
                    )
                    var[i] = reweight(
                        var[i], var_i, num_evals_each1, num_evals_each2[i]
                    )
                    evals_count += evals_count_i

        else:

            # Single round of sampling
            phi = np.zeros(explicand.shape)

            for i in range(self.num_features):
                phi[i], _, evals_count_curr = self._single_feature(
                    i, explicand, baselines, num_evals // self.num_features
                )
                evals_count += evals_count_curr

        return phi, evals_count


class Exact(Estimator):
    """Exact estimation (expoential in the number of features)"""

    def __init__(self, model, num_features):

        super().__init__(model, num_features)

        self.features = set(range(num_features))
        self.max_features = 16
        if self.num_features > self.max_features:

            raise RuntimeError(
                f"Explaining {self.num_features} features is too slow, "
                f"please only use this estimator for <= {self.max_features}"
            )

    def _powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(
            combinations(s, r) for r in range(len(s) + 1)
        )

    def _subset_on_off(self, subset, feature):

        subset_off = np.zeros(self.num_features)
        if subset:
            subset_off[np.array(subset)] = 1

        subset_on = np.copy(subset_off)
        subset_on[feature] = 1

        return (subset_on.astype("bool"), subset_off.astype("bool"))

    def _single_feature(self, feature, explicand, baselines):

        masked_samples = np.repeat(baselines, 2 ** self.num_features, 0)
        sizes = []

        subsets = self._powerset(self.features - set([feature]))
        for i, subset in enumerate(subsets):

            subset_on, subset_off = self._subset_on_off(subset, feature)

            sizes.append(subset_off.sum())

            masked_samples[i, subset_on] = explicand[subset_on]
            masked_samples[
                2 ** (self.num_features - 1) + i, subset_off
            ] = explicand[subset_off]

        # Compute marginal contributions
        weights = 1 / scipy.special.comb(
            self.num_features - 1, np.array(sizes)
        )
        weights /= self.num_features
        preds = self.model(masked_samples)
        preds_on = preds[: 2 ** (self.num_features - 1)]
        preds_off = preds[2 ** (self.num_features - 1) :]
        deltas = weights * (preds_on - preds_off)

        return deltas.sum()

    def _explain(self, explicand, baselines):

        phi = np.zeros(explicand.shape)

        for i in range(self.num_features):
            phi[i] = self._single_feature(i, explicand, baselines)

        return phi

    def __call__(self, explicand, baselines):

        if baselines.shape[0] != 1:
            raise NotImplementedError(
                "Exact estimator only supports a single baseline"
            )

        return self._explain(explicand, baselines)


class MultilinearFeature(FeatureEstimator):
    """Multilinear extension sampling approach (per-feature):

    https://arxiv.org/abs/2010.12082
    """

    def __init__(self, model, num_features):

        self.probs = None

        super().__init__(model, num_features)

    def _sample_subset(self):

        # Random sample if no probs
        prob = np.random.uniform()

        # Trapezoid rule by default
        if self.probs:
            prob = self.probs.pop()

        subset = np.random.binomial(1, prob, size=(self.num_features))

        return subset.astype("bool")

    def _subset_on_off(self, subset, feature):

        subset_on = np.copy(subset)
        subset_on[feature] = True
        subset_off = np.copy(subset)
        subset_off[feature] = False

        return subset_on, subset_off

    def _invert_subset(self, subset):

        return np.invert(subset)

    def __call__(
        self,
        explicand,
        baselines,
        num_evals=100,
        is_adaptive=False,
        is_antithetic=False,
        samples_per_prob=2,
    ):

        # Number of samples per probability (if none, random sampling)
        self.samples_per_prob = samples_per_prob
        self.is_antithetic = is_antithetic
        self.is_adaptive = is_adaptive

        return self._explain(explicand, baselines, num_evals)


class Multilinear(Estimator):
    """Multilinear extension sampling approach:

    https://arxiv.org/abs/2010.12082
    """

    def __init__(self, model, num_features):

        self.probs = None

        super().__init__(model, num_features)
        self.features = np.arange(num_features)

    def _sample_subset(self):

        # Random sample if no probs
        prob = np.random.uniform()

        # Trapezoid rule by default
        if self.probs:
            prob = self.probs.pop()

        subset = np.random.binomial(1, prob, size=(self.num_features))

        return subset.astype("bool")

    def _invert_subset(self, subset):

        return np.invert(subset)

    def _explain(self, explicand, baselines, num_evals, is_antithetic=False):

        # Bookkeeping total number of evaluations
        evals_count = 0

        # Determine the appropriate number of subsets
        num_subsets = num_evals
        if is_antithetic:
            num_subsets //= 2 * (self.num_features + 1)
        else:
            num_subsets //= self.num_features + 1

        # Keep stack of probabilities to draw from for Multilinear
        if self.samples_per_prob:

            adj_num_subsets = num_subsets // self.samples_per_prob

            if adj_num_subsets > 1:

                self.probs = np.arange(0, adj_num_subsets) / (
                    adj_num_subsets - 1
                )
                self.probs = list(np.repeat(self.probs, self.samples_per_prob))

        # Estimate the unnormalized attribution
        phi = np.zeros(explicand.shape)
        for _ in range(num_subsets):

            subset = self._sample_subset()

            # Randomly choose a baseline sample
            baseline_ind = np.random.randint(baselines.shape[0])
            baseline = baselines[baseline_ind]

            # Add inverse subset if antithetic
            subsets = [subset]
            if is_antithetic:
                subsets.append(self._invert_subset(subset))

            for curr_subset in subsets:

                # Evaluate game for known subset
                subset_sample = np.copy(baseline)
                subset_sample[curr_subset] = explicand[curr_subset]
                subset_pred = self.model(subset_sample[None, :])
                evals_count += 1

                subset_samples = np.tile(
                    subset_sample, self.num_features
                ).reshape(self.num_features, -1)

                for i in range(self.num_features):

                    if curr_subset[i]:
                        subset_samples[i, i] = baseline[i]
                    else:
                        subset_samples[i, i] = explicand[i]

                # Compute model predictions at the same time
                subset_preds = self.model(subset_samples)
                evals_count += self.num_features

                for i in range(self.num_features):
                    if curr_subset[i]:
                        phi[i] += subset_pred - subset_preds[i]
                    else:
                        phi[i] += subset_preds[i] - subset_pred

        # Normalize
        if is_antithetic:
            phi = phi / (num_subsets * 2)
        else:
            phi = phi / num_subsets

        return phi, evals_count

    def __call__(
        self,
        explicand,
        baselines,
        num_evals=100,
        is_antithetic=False,
        samples_per_prob=2,
    ):

        # Number of samples per probability (if none, random sampling)
        self.samples_per_prob = samples_per_prob

        return self._explain(explicand, baselines, num_evals, is_antithetic)


class RandomOrderFeature(FeatureEstimator):
    """IME permutation sampling approach per feature:

    https://www.jmlr.org/papers/volume11/strumbelj10a/strumbelj10a.pdf
    """

    def __init__(self, model, num_features):

        super().__init__(model, num_features)
        self.features = np.arange(num_features)

    def _sample_subset(self):

        np.random.shuffle(self.features)

        return self.features

    def _subset_on_off(self, subset, feature):

        feature_ind = np.where(subset == feature)[0][0]
        subset_on = subset[: feature_ind + 1]
        subset_off = subset[:feature_ind]

        return subset_on, subset_off

    def _invert_subset(self, subset):

        return np.flip(subset)

    def __call__(
        self,
        explicand,
        baselines,
        num_evals=100,
        is_adaptive=False,
        is_antithetic=False,
    ):

        self.is_antithetic = is_antithetic
        self.is_adaptive = is_adaptive
        return self._explain(explicand, baselines, num_evals)


class RandomOrder(Estimator):
    """Permutation sampling approach by walking through permutation:

    https://www.sciencedirect.com/science/article/pii/S0305054808000804
    """

    def __init__(self, model, num_features):

        super().__init__(model, num_features)
        self.features = np.arange(num_features)

    def _explain(self, explicand, baselines, num_evals, is_antithetic=False):

        # Bookkeeping total number of evaluations
        evals_count = 0

        # Determine the appropriate number of permutations based on num_evals
        num_permutations = num_evals - 1

        if is_antithetic:
            num_permutations //= 2 * (self.num_features - 1) + 1
        else:
            num_permutations //= self.num_features

        phi = np.zeros(explicand.shape)
        explicand_pred = self.model(explicand[None, :])
        evals_count += 1
        explicand_tiled = np.tile(explicand, self.num_features - 1)
        explicand_tiled = explicand_tiled.reshape(self.num_features - 1, -1)

        for _ in range(num_permutations):

            # Shuffle indices and split subsets based on feature position
            np.random.shuffle(self.features)

            # Randomly choose a baseline sample
            baseline_ind = np.random.randint(baselines.shape[0])
            baseline = baselines[baseline_ind]
            baseline_pred = self.model(baseline[None, :])
            evals_count += 1

            # Create masked samples to evaluate
            masked_samples1 = np.copy(explicand_tiled)

            if is_antithetic:
                masked_samples2 = np.copy(explicand_tiled)

            for i in range(self.num_features - 1):
                subset_forward = self.features[(i + 1) :]
                masked_samples1[i, subset_forward] = baseline[subset_forward]

                if is_antithetic:
                    subset_back = self.features[: -(i + 1)]
                    masked_samples2[i, subset_back] = baseline[subset_back]

            # Get output arrays for both forward and backward
            def _add_subsets(preds):
                return np.hstack([baseline_pred, preds, explicand_pred])

            forward_preds = _add_subsets(self.model(masked_samples1))
            evals_count += len(masked_samples1)
            if is_antithetic:
                backward_preds = _add_subsets(self.model(masked_samples2))
                evals_count += len(masked_samples2)

            # Update estimates of feature attributions
            for i in range(self.num_features):
                feature = self.features[i]
                phi[feature] += forward_preds[i + 1] - forward_preds[i]

                if is_antithetic:
                    feature = self.features[-(i + 1)]
                    phi[feature] += backward_preds[i + 1] - backward_preds[i]

        # Normalize according to whether we used antithetic sampling
        if is_antithetic:
            phi = phi / (num_permutations * 2)
        else:
            phi = phi / num_permutations

        return phi, evals_count

    def __call__(
        self, explicand, baselines, num_evals=100, is_antithetic=False
    ):

        return self._explain(explicand, baselines, num_evals, is_antithetic)


class LeastSquares(Estimator):
    """WLS sampling approach (KernelSHAP):

    https://arxiv.org/abs/1705.07874
    http://proceedings.mlr.press/v130/covert21a/covert21a.pdf
    """

    def __init__(self, model, num_features):

        super().__init__(model, num_features)
        self.features = np.arange(self.num_features)

    def _solve_wls(self, A, b, total):
        """Calculate the regression coefficients."""
        try:
            if len(b.shape) == 2:
                A_inv_one = np.linalg.solve(A, np.ones((self.num_features, 1)))
            else:
                A_inv_one = np.linalg.solve(A, np.ones(self.num_features))
            A_inv_vec = np.linalg.solve(A, b)
            values = A_inv_vec - A_inv_one * (
                np.sum(A_inv_vec, axis=0, keepdims=True) - total
            ) / np.sum(A_inv_one)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Singular matrix inversion, use larger variance_batches"
            )

        return values

    def _explain(self, explicand, baselines, num_evals, is_antithetic=False):

        # Round to the nearest even integer
        num_evals = int(2 * (num_evals // 2))

        # Bookkeeping total number of evaluations
        evals_count = 0

        # Probability of each subset size (from Shapley weighting kernel)
        size_probs = np.arange(1, self.num_features)
        size_probs = 1 / (size_probs * (self.num_features - size_probs))
        size_probs /= np.sum(size_probs)

        # Halve number of subset sizes if we are using paired sampling
        num_subsets = num_evals
        if is_antithetic:
            num_subsets = num_evals // 2

        # Sample appropriate subsets
        subsets = np.zeros((num_evals, self.num_features), dtype=bool)
        subset_sizes = np.random.choice(
            range(1, self.num_features), size=num_subsets, p=size_probs
        )

        # Create masked samples to evaluate output of model
        masked_samples = np.tile(explicand, num_evals).reshape(num_evals, -1)

        # Keep track of explicand output for empty and full game
        explicand_output = self.model(explicand[None, :])
        #     evals_count += 1 # Ignore counts for calculating full and null game
        mean_baseline_output = self.model(baselines).mean()
        #     evals_count += len(baselines)
        # @TODO(hughchen): Figure out how to account for these evaluations

        # Generate masked samples based on random subsets and baselines
        for i, size in enumerate(subset_sizes):
            baseline_ind = np.random.randint(baselines.shape[0])
            baseline = baselines[baseline_ind]
            explicand_inds = np.random.choice(
                self.num_features, size=size, replace=False
            )
            baseline_inds = np.setdiff1d(
                np.arange(self.num_features), explicand_inds
            )

            subsets[i, explicand_inds] = True
            masked_samples[i, baseline_inds] = baseline[baseline_inds]

            if is_antithetic:
                subsets[-(i + 1), baseline_inds] = True
                masked_samples[-(i + 1), explicand_inds] = baseline[
                    explicand_inds
                ]

        outputs = self.model(masked_samples)
        evals_count += len(masked_samples)

        # Calculate intermediate matrices for final calculation
        A_matrix = np.matmul(
            subsets[:, :, np.newaxis].astype(float),
            subsets[:, np.newaxis, :].astype(float),
        )
        b_matrix = (
            subsets.astype(float).T
            * (outputs - mean_baseline_output)[:, np.newaxis].T
        ).T
        A = np.mean(A_matrix, axis=0)
        b = np.mean(b_matrix, axis=0)

        # Calculate shapley value feature attributions based on WLS formulation
        phi = self._solve_wls(A, b, explicand_output - mean_baseline_output)

        return phi, evals_count

    def __call__(
        self, explicand, baselines, num_evals=100, is_antithetic=False
    ):

        return self._explain(explicand, baselines, num_evals, is_antithetic)


class LeastSquaresSGD(Estimator):
    """WLS sampling approach using SGD:

    *Only works for baseline shapley.

    Using pre-existing package for projected gradient descent.  Each iteration
    evaluates model for subset, takes gradient step and then projects based on
    an additive efficient normalization.

    https://hal.inria.fr/hal-03414720/
    """

    def __init__(self, model, num_features, max_label):
        """
        model: model being explained
        num_features: number of features
        max_label: maximum label value
        """

        super().__init__(model, num_features)

        def model_predict(x):
            if len(x.shape) == 1:
                return self.model(x[None, :])
            elif len(x.shape) == 2:
                return self.model(x)

        self.model_predict = model_predict
        self.estimator = SGDshapley(num_features, C=max_label)

    def _explain(self, explicand, baselines, num_evals, is_antithetic=False):

        phi = self.estimator.sgd(
            x=pd.DataFrame(explicand[None, :]).iloc[0],
            fc=self.model_predict,
            ref=pd.DataFrame(baselines).iloc[0],
            n_iter=num_evals,
        ).values

        return phi, num_evals

    def __call__(self, explicand, baselines, num_evals=100):

        if baselines.shape[0] != 1:
            raise NotImplementedError(
                "LeastSquaresSGD only supports a single baseline"
            )

        return self._explain(explicand, baselines, num_evals)
