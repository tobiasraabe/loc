#!/usr/bin/env python

"""This module delivers descriptive statistics for panel.pkl.

"""

import json
import pandas as pd
from bld.project_paths import project_paths_join as ppj


CONTROLS = ['AGE', 'GENDER',
            'HH_NET_INCOME_YEAR', 'MARITAL_STATUS',
            'MIGRATION_STATUS', 'EDUCATION_GROUPS_ISCED97']


EVENT_VARIABLES = ['CHILD_DISORDER', 'DEATH_CHILD', 'DEATH_FATHER',
                   'DEATH_HH_PERSON', 'DEATH_MOTHER', 'DEATH_PARTNER',
                   'DIVORCED', 'HH_COMP_CHANGE', 'LAST_JOB_ENDED',
                   'LAST_JOB_ENDED_LIMITED', 'LEGALLY_HANDICAPPED_PERC',
                   'PREGNANCY_UNPLANNED', 'SEPARATED']

DESCRIPTIVES = {'AGE': 'Age',
                'HH_NET_INCOME_YEAR': 'Net yearly wage',
                'MIGRATION_STATUS': 'Migration status',
                'GENDER_Female': 'Female',
                'MARITAL_STATUS_Relationship': 'Married',
                'EDUCATION_GROUPS_ISCED97_[0] in school': 'In school',
                'EDUCATION_GROUPS_ISCED97_[1] inadequately': 'Other',
                'EDUCATION_GROUPS_ISCED97_[2] general elementary':
                'General elementary',
                'EDUCATION_GROUPS_ISCED97_[3] middle vocational':
                'Middle vocational',
                'EDUCATION_GROUPS_ISCED97_[4] vocational + Abi':
                'Vocational + Abi',
                'EDUCATION_GROUPS_ISCED97_[5] higher vocational':
                'Higher vocational',
                'EDUCATION_GROUPS_ISCED97_[6] higher education':
                'Higher education',
                }

RETAINED_COLUMNS = ['ID', 'YEAR'] + CONTROLS
RETAINED_COLUMNS += ['EVENT_' + i for i in EVENT_VARIABLES]


def prepare_data():
    # Load data
    df_2005_2010 = pd.read_pickle(
        ppj('OUT_DATA', 'panel_2005_2010_inspection.pkl'))
    df_2005_2010 = df_2005_2010[RETAINED_COLUMNS]
    df_2010_2015 = pd.read_pickle(
        ppj('OUT_DATA', 'panel_2010_2015_inspection.pkl'))
    df_2010_2015 = df_2010_2015[RETAINED_COLUMNS]
    # Create identifiers for the whole period whether an event is experienced
    # by an individual
    for df in [df_2005_2010, df_2010_2015]:
        for i in EVENT_VARIABLES:
            df['EVENT_' + i] = df.groupby('ID')['EVENT_' + i].transform(
                lambda x: x.notnull().any())
    # Get the starting period of the individual which is in 2005 or 2010
    df_2005_2010 = df_2005_2010.groupby('ID', as_index=False).first()
    df_2010_2015 = df_2010_2015.groupby('ID', as_index=False).first()
    # Append the two frame
    df = df_2005_2010.append(df_2010_2015)

    # Create identifier for all events
    df['EVENT_ANY'] = df[['EVENT_' + i for i in EVENT_VARIABLES]].any(
        axis='columns')

    # Select only ids from the estimation sample
    panel = pd.read_pickle(ppj('OUT_DATA', 'panel.pkl'))
    ids = panel.ID.unique()
    df = df.loc[df.ID.isin(ids)].copy()

    return df


def generate_numbers(df):
    # Generate numbers of observations
    gen_num = {}
    gen_num['Number of observations'] = df.shape[0]
    gen_num['Number of unique IDs'] = df.ID.unique().shape[0]
    gen_num['Number of observations in both periods'] = (
        gen_num['Number of observations'] -
        gen_num['Number of unique IDs'])

    with open(ppj('OUT_TABLES',
                  'panel_descriptives_numbers.json'), 'w') as file:
        file.write(json.dumps(gen_num, indent=4, sort_keys=True))


def generate_descriptive_statistics(df, event):
    """This function creates a big table with all sample statistics for the
    groups who have experienced an event and the ones not."""

    # Define names
    event_col = 'EVENT_' + event
    event_name = event.replace('_', ' ').lower()
    # Create dataframe with control variables and event variable
    df = pd.concat([pd.get_dummies(df[CONTROLS]), df[event_col]],
                   axis=1)
    # Save columns names
    col_names = list(df.drop(event_col, axis=1).columns)
    # Create table
    table = df.groupby(event_col)[col_names].mean().T
    table.drop(['GENDER_Male', 'MARITAL_STATUS_Single'], axis='rows',
               inplace=True)
    table = table.rename(index=DESCRIPTIVES)

    event_fn = event.replace('_', '-').lower()
    with open(
        ppj('OUT_TABLES',
            f'tab-panel-descriptive-statistics-{event_fn}.tex'), 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write(
            f'\t\\caption{{Descriptive Statistics - Event: {event_name}}}\n')
        f.write(f'\t\\label{{tab:descriptive-statistics-{event_fn}}}\n')
        f.write('\t\\centering')
        f.write('\t\\begin{tabular}{>{\quad}lcc}\n')
        f.write('\t\\toprule\n')
        f.write(
            f'\t & \\textbf {{No Event}} & \\textbf{{Event}} \\\\\n')
        f.write('\t & mean/share & mean/share \\\\\n')
        f.write('\t\\midrule\n')

        for variable, [event_false, event_true] in table.iterrows():
            if variable in ['In school']:
                f.write('\t\\rule{0pt}{2.5ex}')
                f.write(
                    '\t\\rowgroup{\\textit{School degree}} \\\\\n')
            f.write(
                f'\t{variable} & {event_false:.3f} & {event_true:.3f} \\\\\n')

        f.write('\t\\bottomrule\n')
        f.write('\t\\end{tabular}\n')
        f.write('\\end{table}\n')


if __name__ == '__main__':
    # Prepare data
    df = prepare_data()
    # Generate numbers and write to json
    generate_numbers(df)
    # Generate table of descriptive statistics
    for event in ['ANY'] + EVENT_VARIABLES:
        generate_descriptive_statistics(df, event)
