import sys

import matplotlib.pyplot as plt
import numpy as np
from bld.project_paths import project_paths_join as ppj
from sklearn.externals import joblib


def plot(model, model_type):
    """Plot the cumulative explained variance of a pca model."""
    pca = model.named_steps["pca"]

    # Begin figure
    fig, ax = plt.subplots()

    ax.plot(pca.explained_variance_ratio_, label="Single")
    ax.plot(np.cumsum(pca.explained_variance_ratio_), label="Cumulativ")

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Explained Variance Ratio")

    ax.set_ylim(0, 1.1)

    ax.legend()

    name = model_type.replace("_", "-")
    plt.savefig(ppj("OUT_FIGURES", f"fig-pca-{name}-explained-variance.png"))


if __name__ == "__main__":
    # Load model type
    model_type = sys.argv[1]
    # Load the model
    container = joblib.load(ppj("OUT_DATA", f"model_pca_{model_type}.pkl"))
    model = container["model"]

    plot(model, model_type)
