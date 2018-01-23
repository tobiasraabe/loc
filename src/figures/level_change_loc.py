#!/usr/bin/env python

import pandas as pd
from bld.project_paths import project_paths_join as ppj
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

EVENT_VARIABLES = ['CHILD_DISORDER', 'DEATH_CHILD', 'DEATH_FATHER',
                   'DEATH_HH_PERSON', 'DEATH_MOTHER', 'DEATH_PARTNER',
                   'DIVORCED', 'HH_COMP_CHANGE', 'LAST_JOB_ENDED',
                   'LEGALLY_HANDICAPPED_PERC',
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
    'EVENT_LEGALLY_HANDICAPPED_PERC': 'Legally Handicapped',
    'EVENT_PREGNANCY_UNPLANNED': 'Unplanned Pregnancy',
    'EVENT_SEPARATED': 'Separation'
}


def prepare_data():
    # Load data and restrict to columns
    loc = pd.read_pickle(ppj('OUT_DATA', 'loc_fa.pkl'))
    loc = loc[['ID', 'YEAR', 'FIRST_FACTOR_DELTA']]
    # Load data and restrict to columns
    df = pd.read_pickle(ppj('OUT_DATA', 'panel.pkl'))
    # Create identifier whether an event was experienced
    df['EVENT_ANY'] = df[['EVENT_' + i for i in EVENT_VARIABLES]].any(
        axis='columns')
    df = df[['ID', 'YEAR', 'EVENT_ANY'] +
            ['EVENT_' + i for i in EVENT_VARIABLES]]

    df = df.merge(loc, on=['ID', 'YEAR'], how='left')

    table = pd.DataFrame(index=[False, True])
    for i in EVENT_VARIABLES:
        table = pd.concat([table, df.groupby(
            'EVENT_' + i).FIRST_FACTOR_DELTA.mean()], axis=1)
        table = table.rename(columns={'FIRST_FACTOR_DELTA': 'EVENT_' + i})
        model = smf.ols('FIRST_FACTOR_DELTA ~ EVENT_' + i, df).fit(
            cov_type='cluster', cov_kwds={'groups': df.ID})
        table.loc['se_cons', 'EVENT_' + i] = model.bse[0]
        table.loc['se', 'EVENT_' + i] = model.bse[1]

    # Rename columns and index
    table = table.rename(index={False: 'No event', True: 'Event'},
                         columns=EVENT_MAP)

    return table.T


def create_graph(table):
    fig, ax = plt.subplots()

    table.iloc[:, :2].plot.bar(rot=70, ax=ax)

    ax.set_ylabel('$\Delta$ Stated locus of control')

    ax.legend()

    plt.savefig(ppj('OUT_FIGURES', 'fig-level-change-loc-event.png'))


if __name__ == '__main__':
    table = prepare_data()
    create_graph(table)
