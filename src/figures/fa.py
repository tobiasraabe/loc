#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

from bld.project_paths import project_paths_join as ppj
from sklearn.externals import joblib


def plot(model, model_type):
    """Plot the cumulative explained variance of a fa model."""
    fa = model.named_steps['fa']

    # Calculate explained variance ratio
    m1 = fa.components_ ** 2
    m2 = np.sum(m1, axis=1)
    explained_variance_ratio = m2 / np.sum(m2)

    # Begin figure
    fig, ax = plt.subplots()

    ax.set_xlabel('Number of Factors')
    ax.set_ylabel('Explained Variance Ratio')

    ax.plot(explained_variance_ratio, label='Single')
    ax.plot(np.cumsum(explained_variance_ratio), label='Cumulativ')

    ax.legend()

    plt.tight_layout()

    name = model_type.replace('_', '-')
    plt.savefig(ppj('OUT_FIGURES', f'fig-fa-{name}-explained-variance.png'))


if __name__ == '__main__':
    # Load model type
    model_type = sys.argv[1]
    # Load the model
    container = joblib.load(ppj('OUT_DATA', f'model_fa_{model_type}.pkl'))
    model = container['model']

    plot(model, model_type)
