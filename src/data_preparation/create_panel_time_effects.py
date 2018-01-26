#!/usr/bin/env python

import pandas as pd
from bld.project_paths import project_paths_join as ppj


EVENT_VARIABLES = ['CHILD_DISORDER', 'DEATH_CHILD', 'DEATH_FATHER',
                   'DEATH_HH_PERSON', 'DEATH_MOTHER', 'DEATH_PARTNER',
                   'DIVORCED', 'HH_COMP_CHANGE', 'LAST_JOB_ENDED',
                   'LAST_JOB_ENDED_LIMITED', 'LEGALLY_HANDICAPPED_PERC',
                   'PREGNANCY_UNPLANNED', 'SEPARATED']


def restrict_sample(df):
    """This function restricts the sample in the following way:
           1. Only observations observed from 2005-2015
           2. Only observations without events in 2005-2010
           3. Only the period 2010-2015
    """
    # 1. Restriction
    set_id_2010 = set(df.loc[df.YEAR == 2010, 'ID'].unique())
    set_id_2015 = set(df.loc[df.YEAR == 2015, 'ID'].unique())
    # Intersection of IDs
    set_id = set_id_2010 & set_id_2015
    # Select valid observations
    df = df.loc[df.ID.isin(set_id)].copy()

    # 2. and 3. Restriction
    df['EVENT_ALL_COUNT'] = df[
        ['EVENT_' + i + '_COUNT' for i in EVENT_VARIABLES]].sum(axis=1)
    valid_ids = df.loc[(df.YEAR == 2010) &
                       (df.EVENT_ALL_COUNT == 0), 'ID'].values
    df = df.loc[(df.YEAR == 2015) & (df.ID.isin(valid_ids))]

    list_counts = [i for i in df if 'EVENT' in i and 'COUNT' in i and
                   'PREVIOUS' not in i and 'LIMITED' not in i and 'LEADS' not
                   in i]
    # Set up column with counts of all traumatic events.
    df['EVENT_COUNTS_ALL'] = df[list_counts].sum(axis=1)

    return df


if __name__ == '__main__':
    # Load dataframe
    df = pd.read_pickle(ppj('OUT_DATA', 'panel.pkl'))
    # Restrict sample
    df = restrict_sample(df)
    # Save
    df.to_pickle(ppj('OUT_DATA', 'panel_time_effects.pkl'))
