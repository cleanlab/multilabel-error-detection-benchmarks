# Benchmarking label error detection algorithms for multi-label classification

Code to reproduce results from the paper:

[**Identifying Incorrect Annotations in Multi-Label Classification Data**](https://arxiv.org/abs/2211.13895)

This package is a DVC project that uses various datasets to evaluate different label quality scores for detecting annotation errors in multi-label classification.

## Instructions to get started

1. Clone the repo
2a [*Optional*]. Open the repo in a devcontainer
2b. Install the requirements with:
```bash
pip install -r requirements.txt
```
3. Run the pipeline with:

```bash
dvc repro
```

  - The pipeline has several stages:

  ```bash
  $ dvc dag
                +--------------+
                | make_dataset |
                +--------------+
                 ***          ***
                *                *
              **                  **
  +------------------+          +-------+
  | get_avg_accuracy |          | train |
  +------------------+          +-------+
            *                        *
            *                        *
            *                        *
    +-------------+         +---------------+
    | group_stats |         | score_classes |
    +-------------+         +---------------+
                                     *
                                     *
                                     *
                              +-----------+
                              | aggregate |
                              +-----------+
                                     *
                                     *
                                     *
                             +--------------+
                             | rank_metrics |
                             +--------------+
                                     *
                                     *
                                     *
                             +--------------+
                             | plot_metrics |
                             +--------------+
  +----------------+
  | plot_avg_trace |
  +----------------+
  ```

  A description of each stage is given below.
  ```
  $ dvc stage list
  make_dataset      Create groups of datasets of different sizes & number of classes.
  train             Train models and get out-of-sample predicted probabilities on the training sets.
  get_avg_accuracy  Get model performance metrics on test sets, with and without label errors.
  group_stats       Summarize model performance metrics for each group of datasets.
  score_classes     Compute class label quality scores for each example in a dataset.
  aggregate         Aggregate class label quality scores for all classes into a single score.
  rank_metrics      Compute label error detection metrics for aggregated scores.
  plot_metrics      Plot the label error detection and ranking metrics for the aggregated scores.
  plot_avg_trace    Plot average traces of noise matrices used for noisy label generation.
  ```

  - The `group_stats` stage outputs two files in `data/accuracies/`:
    - `results_group.csv`: All experimental results
    - `results_agg.json`: Overall stats for the different aggregator methods.

  - The stages have variouus output files and directories. This is best viewed with `dvc dag -o`. Ignoring most of the intermediate files, the most relevant files are:
    - data/accuracy/results_group.csv: Statistics of model performance metrics for each group of datasets.
    - data/scores/results.csv: Class label quality scores for each example in each dataset.
    - data/scores/metrics.csv: Statistics of label error detection and ranking metrics for each group of datasets.


4. Inspect the synthetic datasets in the `notebooks/inspect_generated_data.ipynb` notebook.
5. Inspect the results in the `notebooks/inspect_score_results.ipynb` notebook.

## Aggregation methods to pool per-class annotation scores into an overall label quality score for each example

Along with the typical `np.mean`, `np.median`, `np.min`, `np.max` aggregators, we also implement several methods found in `src/evaluation/aggregate.py`:

- `softmin_pooling`
- `log_transform_pooling`
- `cumulative_average`
- `simple_moving_average`
- `exponential_moving_average`
- `weighted_cumulative_average`

## CelebA analysis

See the [Examples Notebooks](https://github.com/cleanlab/examples/tree/master/multilabel_classification) in our [examples](https://github.com/cleanlab/examples/) repository for:

- the Pytorch code we used to train a multi-label classifier model on CelebA
- the code to find mislabeled images in this dataset
