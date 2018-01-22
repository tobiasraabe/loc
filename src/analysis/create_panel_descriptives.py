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
    # Load dataframe
    df = pd.read_pickle(ppj('OUT_DATA', 'panel.pkl'))
    # Generate numbers and write to json
    generate_numbers(df)
    # Generate table of descriptive statistics
    for event in EVENT_VARIABLES:
        generate_descriptive_statistics(df, event)
