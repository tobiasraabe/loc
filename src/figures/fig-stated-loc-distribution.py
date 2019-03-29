import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from bld.project_paths import project_paths_join as ppj


def plot_complete_distribution(first_factor, decomposition):
    x = np.linspace(first_factor.min(), first_factor.max(), 1000)

    fig, ax = plt.subplots()

    ax.hist(first_factor, bins=40, width=0.1, density=True)

    if decomposition in ["fa"]:

        ax.plot(x, norm.pdf(x))
        ax.set_xlim(-3.5, 3.5)
        ax.set_xticks(list(range(-3, 4)))
        ax.set_xticklabels(["External"] + list(range(-2, 3)) + ["Internal"])

    ax.set_xlabel("Locus of Control")
    ax.set_ylabel("Density")

    plt.savefig(ppj("OUT_FIGURES", f"fig-stated-loc-{decomposition}.png"))


def plot_distribution_by_year(df, decomposition):
    x = np.linspace(df.FIRST_FACTOR.min(), df.FIRST_FACTOR.max(), 1000)

    fig, ax = plt.subplots()

    df.groupby("YEAR").FIRST_FACTOR.plot.kde(ax=ax, alpha=0.7)

    if decomposition in ["fa"]:

        ax.plot(
            x,
            norm.pdf(x),
            color="grey",
            ls="--",
            label="Standard Normal Distribution",
            alpha=0.7,
        )
        ax.set_xlim(-3.5, 3.5)
        ax.set_xticks(list(range(-3, 4)))
        ax.set_xticklabels(["External"] + list(range(-2, 3)) + ["Internal"])

    ax.set_xlabel("Locus of Control")
    ax.set_ylabel("Density")

    ax.legend()

    plt.savefig(ppj("OUT_FIGURES", f"fig-stated-loc-by-year-{decomposition}.png"))


def main():
    decomposition = sys.argv[1]

    df = pd.read_pickle(ppj("OUT_DATA", f"loc_{decomposition}.pkl"))

    plot_complete_distribution(df.FIRST_FACTOR.values, decomposition)

    plot_distribution_by_year(df, decomposition)


if __name__ == "__main__":
    main()
