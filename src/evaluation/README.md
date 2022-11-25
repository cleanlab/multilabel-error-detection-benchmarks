# Score aggregation benchmark

This package contains the code to aggregate the class label quality scores for each example in a dataset, and to compute label error detection metrics for the aggregated scores.

- [class_score.py](./class_score.py) : Computes class label quality scores for each example in a dataset.
  - Called with `dvc repro score_classes` from the root directory of the repository.
- [score.py](./score.py) : Aggregates class label quality scores into a single score for each example in a dataset.
  - Depends on [aggregate.py](./aggregate.py) which defines the aggregation methods.
  - Called with `dvc repro aggregate` from the root directory of the repository.
- [eval_ranking_metrics.py](./eval_ranking_metrics.py) : Computes label error detection metrics for aggregated scores.
  - Called with `dvc repro rank_metrics` from the root directory of the repository.
- [plot_metrics.py](./plot_metrics.py) : Plots the label error detection and ranking metrics for the aggregated scores.
  - Called with `dvc repro plot_metrics` from the root directory of the repository.
