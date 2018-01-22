#!/usr/bin/env python

import pandas as pd
from bld.project_paths import project_paths_join as ppj


def prepare(df, loc):
    # Load data
    df = pd.read_pickle(ppj('OUT_DATA', 'panel.pkl'))
    loc_data = pd.read_pickle(ppj('OUT_DATA', 'loc_pca.pkl'))

    # Separate loc data for the three different years.
    loc_years = [2005, 2010, 2015]
    loc_data_2005, loc_data_2010, loc_data_2015 = [
        loc_data.loc[loc_data['YEAR'] == i] for i in loc_years]

    # Calculate differences of first factor by substracting the previous from
    # the following period
    loc_data.sort_values(['ID', 'YEAR'], axis='rows', inplace=True)
    loc_data['FIRST_FACTOR_DELTA'] = loc_data.groupby(
        'ID')['FIRST_FACTOR'].transform(pd.Series.diff)

    # Merge FIRST_FACTOR_DELTA in panel
    df = df.merge(loc_data[['ID', 'YEAR', 'FIRST_FACTOR_DELTA']],
                  on=['ID', 'YEAR'], how='left')

    # Form 8 age groups. Each group encompasses about one decade.
    df['AGE_GROUPS'] = pd.cut(df['AGE'], list(range(20, 81, 10)) + [105])

    # Cut the time differences into yearly intervals.
    list_time_diffs = [i for i in df if 'DIFF' in i]
    for item in list_time_diffs:
        df[item + '_GROUP'] = pd.cut(df[item], list(range(0, 73, 12)),
                                     include_lowest=True)

    # Save the dataframe which is needed for the regressions as a pickle.
    df.to_pickle(ppj('OUT_ANALYSIS', 'panel_reg.pkl'))


if __name__ == '__main__':
    loc = pd.read_pickle(ppj('OUT_DATA', 'loc_pca.pkl'))
    df = pd.read_pickle(ppj('OUT_DATA', 'panel.pkl'))
    prepare(df, loc)
