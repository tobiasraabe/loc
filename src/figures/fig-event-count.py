#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj


EVENT_VARIABLES = ['CHILD_DISORDER', 'DEATH_CHILD', 'DEATH_FATHER',
                   'DEATH_HH_PERSON', 'DEATH_MOTHER', 'DEATH_PARTNER',
                   'DIVORCED', 'HH_COMP_CHANGE', 'LAST_JOB_ENDED',
                   'LAST_JOB_ENDED_LIMITED', 'LEGALLY_HANDICAPPED_PERC',
                   'PREGNANCY_UNPLANNED', 'SEPARATED']

EVENT_MAP = {
    'EVENT_CHILD_DISORDER': 'Child has Disorders',
    'EVENT_DEATH_CHILD': 'Death of Child',
    'EVENT_DEATH_FATHER': 'Death of Father',
    'EVENT_DEATH_HH_PERSON': 'Death of HH Person',
    'EVENT_DEATH_MOTHER': 'Death of Mother',
    'EVENT_DEATH_PARTNER': 'Death of Partner',
    'EVENT_DIVORCED': 'Divorce',
    'EVENT_HH_COMP_CHANGE': 'HH Composition Change',
    'EVENT_LAST_JOB_ENDED': 'Displacement',
    'EVENT_LAST_JOB_ENDED_LIMITED': 'Displacement (restricted)',
    'EVENT_LEGALLY_HANDICAPPED_PERC': 'Legally Handicapped',
    'EVENT_PREGNANCY_UNPLANNED': 'Unplanned Pregnancy',
    'EVENT_SEPARATED': 'Separation'
}


def main():
    df = pd.read_pickle(ppj('OUT_DATA', 'panel.pkl'))

    fig, axs = plt.subplots(4, 4, figsize=(16, 16), sharex=False)

    axs = axs.flatten()

    for i, event in enumerate(EVENT_VARIABLES):
        data = df['EVENT_' + event + '_COUNT'].value_counts()
        data.plot.bar(ax=axs[i])
        for x, y in data.iteritems():
            axs[i].annotate(xy=(x - 0.15, y + 500), s=y)

        axs[i].set_xticks(list(range(6)))
        axs[i].set_xticklabels(list(range(6)))

        axs[i].set_xlabel('Number of ' + EVENT_MAP['EVENT_' + event])

        axs[i].set_ylim(0, 25000)

    for i in range(13, 16):
        axs[i].axis('off')

    plt.savefig(ppj('OUT_FIGURES', 'fig-event-count.png'))


if __name__ == '__main__':
    main()
