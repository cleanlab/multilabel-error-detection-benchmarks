# Data generation

This package contains the code to generate the data used in label error detection benchmark for multi-label classification.

- [make_dataset.py](./make_dataset.py) : Generates the datasets used in the benchmark, and saves them in the `data/generated/
` directory.
  - Called with `dvc repro make_dataset` from the root directory of the repository.
- [plot_avg_trace.py](./plot_avg_trace.py) : Plots an example of average traces of noise matrices used for noisy label generation.

  **NOTE**: This plot is unrelated to the benchmark, and is only included for demonstration purposes.

  - Called with `dvc repro plot_avg_trace` from the root directory of the repository.
