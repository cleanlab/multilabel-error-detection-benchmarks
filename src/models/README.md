# Model training and out-of-sample evaluation of predicted probabilities

This package contains the code to train the models used in the benchmark, and to compute out-of-sample predicted probabilities on the training sets generated with the code in the [`data`](../data/README.md) package.

## Label error benchmarks
- [train.py](./train.py) : Trains the models and saves the out-of-sample predicted probabilities in the `data/pred_probs/` directory.
  - Called with `dvc repro train` from the root directory of the repository.


## Model performance benchmarks
- [avg_accuracy.py](./avg_accuracy.py) : Computes model performance metrics on test sets, with and without label errors.
  - Called with `dvc repro get_avg_accuracy` from the root directory of the repository.
- [group_stats.py](./group_stats.py) : Summarizes model performance metrics for each group of datasets.
  - Called with `dvc repro group_stats` from the root directory of the repository.


