#!/usr/bin/env python

import pandas as pd
from bld.project_paths import project_paths_join as ppj
import matplotlib.pyplot as plt
import numpy as np


def plot_loadings():
    fa = pd.read_pickle(ppj('OUT_DATA', 'fa_loadings.pkl'))

    fig, ax = plt.subplots()

    ax.axhline(y=0, color='grey', alpha=0.7)
    ax.axvline(x=0, color='grey', alpha=0.7)

    ax.plot(fa.FIRST_FACTOR, fa.SECOND_FACTOR, 'o')

    for i, [x, y] in fa.iterrows():
        if i in ['LOC_DOUBT', 'LOC_ACHIEVED_DESERVE']:
            y += 0.07
        elif i in ['LOC_LITTLE_CONTROL', 'LOC_OTHERS']:
            y -= 0.12
        elif i in ['LOC_POSSIBILITIES']:
            y -= 0.09
        ax.annotate(i.replace('_', ' ').title(), xy=(x + 0.05, y))

    ax.set_xlabel('First Factor')
    ax.set_ylabel('Second Factor')

    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_yticks(np.arange(-1, 1.1, 0.5))

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    plt.savefig(ppj('OUT_FIGURES', 'fig-fa-factor-loadings.png'))


if __name__ == '__main__':
    plot_loadings()
