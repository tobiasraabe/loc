import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join as ppj


LOC_MAP = {
    "LOC_LUCK": "Item 3",
    "LOC_ACHIEVED_DESERVE": "Item 2",
    "LOC_POSSIBILITIES": "Item 8",
    "LOC_LIFES_COURSE": "Item 1",
    "LOC_DOUBT": "Item 7",
    "LOC_OTHERS": "Item 5",
    "LOC_LITTLE_CONTROL": "Item 10",
}


def plot_loadings():
    fa = pd.read_pickle(ppj("OUT_DATA", "fa_loadings.pkl"))

    fig, ax = plt.subplots()

    ax.axhline(y=0, color="grey", alpha=0.7)
    ax.axvline(x=0, color="grey", alpha=0.7)

    ax.plot(fa.FIRST_FACTOR, fa.SECOND_FACTOR, "o")

    for i, [x, y] in fa.iterrows():
        if i in ["LOC_LITTLE_CONTROL", "LOC_OTHERS"]:
            y -= 0.06
        elif i in ["LOC_ACHIEVED_DESERVE"]:
            y += 0.01
        elif i in ["LOC_POSSIBILITIES"]:
            y -= 0.09
        ax.annotate(LOC_MAP[i], xy=(x - 0.02, y), ha="right")

    ax.set_xlabel("First Factor")
    ax.set_ylabel("Second Factor")

    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_yticks(np.arange(-1, 1.1, 0.5))

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    plt.savefig(ppj("OUT_FIGURES", "fig-fa-factor-loadings.png"))


if __name__ == "__main__":
    plot_loadings()
