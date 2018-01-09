#!/usr/bin/env python

import pandas as pd
from bld.project_paths import project_paths_join as ppj


LOC_VALUES = {'[1] Trifft ueberhaupt nicht zu': 1, '[2] [2/10]': 2,
              '[3] [3/10]': 3, '[4] [4/10]': 4, '[5] [5/10]': 5,
              '[6] [6/10]': 6, '[7] Trifft voll zu': 7}


def clean_variables(df):
    # Replace values and cast to integers
    for variable in df.select_dtypes('category'):
        df[variable].cat.rename_categories(LOC_VALUES, inplace=True)
        df[variable] = pd.to_numeric(df[variable], errors='raise',
                                     downcast='integer')
    return df


if __name__ == '__main__':
    # Load dataset
    df = pd.read_pickle(ppj('OUT_DATA', 'loc_raw.pkl'))
    # Clean the data
    df = clean_variables(df)
    # Save data
    df.to_pickle(ppj('OUT_DATA', 'loc.pkl'))
