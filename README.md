# Benchmarking label error detection algorithms for multi-label classification

A DVC project for running benchmarks on the quality of label scores for multi-label classification with synthetic data.

## Instructions

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

  - The pipeline has 3 stages:
  
  ```bash
  $ dvc dag
          +--------------+                
          | make_dataset |                
          +--------------+                
           ***         ***                
          *               *               
        **                 **             
  +-------+         +------------------+  
  | train |         | get_avg_accuracy |  
  +-------+         +------------------+  
      *                       *           
      *                       *           
      *                       *           
  +-------+           +-------------+     
  | score |           | group_stats |     
  +-------+           +-------------+     
  +----------------+ 
  | plot_avg_trace | 
  +----------------+ 
  ```
    
  - The `score` stage outputs two files in `data/scores`:
    - `results.csv`: All experimental results
    - `results_agg.json`: Overall stats for the different aggregator methods.

4. Inspect the synthetic datasets in the `notebooks/inspect_generated_data.ipynb` notebook.
5. Inspect the results in the `notebooks/inspect_score_results.ipynb` notebook.

## Aggregator methods

Along with the typical `np.mean`, `np.median`, `np.min`, `np.max` aggregators, we also implement several methods found in `src/evaluation/aggregate.py`:

- `softmin_pooling`
- `log_transform_pooling`
- `cumulative_average`
- `simple_moving_average`
- `exponential_moving_average`
- `weighted_cumulative_average`