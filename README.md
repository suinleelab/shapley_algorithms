# Shapley Algorithms

Algorithms to estimate Shapley value feature attributions.  

## Installation

```bash
git clone https://github.com/suinleelab/shapley_algorithms.git
cd shapley_algorithms
pip install -e .
```

## Explanation code

Implementations of a number of algorithms based on the random order, least squares, and multilinear extension characterizations of Shapley values.  In addition, we include variants on each approach: adaptive and antithetic sampling.  

Algorithms are implemented here: `shapley_algorithms/explain.py`

## Datasets

- [Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- [NHANES](https://www.cdc.gov/nchs/nhanes/index.htm)
- [Blog](https://archive.ics.uci.edu/ml/datasets/BlogFeedback)

Datasets are loaded here: `shapley_algorithms/data.py`

## Benchmarking

Benchmarking is performed for each dataset (and cached locally) in `notebooks`.

<!-- ## Citation

If you use any part of this code and pretrained weights for your own purpose, please cite
our [paper](https://arxiv.org/abs/2206.05282). -->

## Contact

- [Hugh Chen](http://hughchen.github.io/) (Paul G. Allen School of Computer Science and Engineering @ University of
  Washington)
- [Ian Covert](https://iancovert.com) (Paul G. Allen School of Computer Science and Engineering @ University of
  Washington)
- [Scott Lundberg](https://scottlundberg.com/) (Microsoft Research)
- [Su-In Lee](https://suinlee.cs.washington.edu/) (Paul G. Allen School of Computer Science and Engineering @ University
  of Washington)
