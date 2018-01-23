#!/usr/bin/env python

import sys
import pandas as pd
from bld.project_paths import project_paths_join as ppj
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


def prepare_data(decomposition):
    df = pd.read_pickle(ppj('OUT_DATA', f'loc_{decomposition}.pkl'))

    return df.FIRST_FACTOR


def create_graph(s, decomposition):
    # Prepare normal distribution
    x = np.linspace(s.min(), s.max(), 1000)

    fig, ax = plt.subplots()

    ax.hist(s, bins=40, width=0.1, density=True)

    if decomposition in ['fa']:
        ax.plot(x, norm.pdf(x, loc=0, scale=1))
        ax.set_xlim(-3.5, 3.5)
        ax.set_xticks(list(range(-3, 4)))
        ax.set_xticklabels(['External'] + list(range(-2, 3)) + ['Internal'])
    elif decomposition in ['pca']:
        pass

    ax.set_xlabel('Stated locus of control')
    ax.set_ylabel('Density')

    plt.savefig(ppj('OUT_FIGURES', f'fig-stated-loc-{decomposition}.png'))


if __name__ == '__main__':
    decomposition = sys.argv[1]
    # Prepare the data
    s = prepare_data(decomposition)
    # Create histogram
    create_graph(s, decomposition)
