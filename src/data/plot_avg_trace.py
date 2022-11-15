import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


def plot_avg_trace_sample(K: int, shape: float, scale: float):
    np.random.seed(0)
    avg_traces = 1 - np.random.gamma(shape, scale, size=K)
    weights = (1 - (np.exp(-np.arange(K)**2/K)/(2*K)))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=1200)
    X = 1 + np.arange(K)

    # Plot the samples and weights
    ax.scatter(X, np.sort(avg_traces), alpha=0.5,
               color='c', label='$(1 - X_k)$ (sorted)')
    ax.plot(X, np.sort(weights), alpha=0.3, label='Weights')

    # Apply the weights
    avg_traces[np.argsort(avg_traces)] *= weights

    # Plot the weighted samples
    ax.scatter(X, np.sort(avg_traces), alpha=0.5, color='r', label=r'$Y_k$ (sorted)')

    # Set figure properties
    ax.set_ylim(min(np.min(avg_traces), np.min(weights))-scale, 1+scale)
    ax.set_xlabel(r'$k$', fontsize=16)
    ax.legend(fontsize=14)

    return fig, ax


def main(K: int, shape: float, scale: float):

    fig, _ = plot_avg_trace_sample(K, shape, scale)

    IMAGE_DIR = Path("data/images")
    IMAGE_DIR.mkdir(exist_ok=True)

    figname = "avg_trace.svg"
    # Save the figure as a .svg
    fig.savefig(IMAGE_DIR / figname, bbox_inches='tight')


if __name__ == "__main__":
    # Load params.yaml
    all_parms = yaml.load(open("params.yaml"), Loader=yaml.FullLoader)
    small_dataset_kwargs = all_parms["dataset_kwargs"]["small"]
    gamma_params = small_dataset_kwargs.pop("gamma")

    # Set main arguments
    shape, scale = gamma_params["shape"], gamma_params["scale"]
    K = 10

    # Plot and save
    main(K=K, shape=shape, scale=scale)
