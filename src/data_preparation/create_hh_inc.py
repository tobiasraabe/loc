#!/usr/bin/env python

"""Prepare ``hgen.dta`` to extract monthly household net income.

"""

import pandas as pd
import numpy as np
from bld.project_paths import project_paths_join as ppj


VARIABLE_NAMES_HGEN = {
    'hid': 'ID_HH',
    'syear': 'YEAR',
    'hghinc': 'HH_NET_INCOME_MONTHLY',
}

RETAINED_COLUMNS_HGEN = list(VARIABLE_NAMES_HGEN.keys())


def main():
    # Load dataset
    df = pd.read_stata(ppj('IN_DATA', 'hgen.dta'),
                       columns=RETAINED_COLUMNS_HGEN)
    df = df.rename(columns=VARIABLE_NAMES_HGEN)
    # Cast negative values to NaNs
    df.loc[df.HH_NET_INCOME_MONTHLY < 0, 'HH_NET_INCOME_MONTHLY'] = np.nan
    # Drop NaNs
    df.dropna(subset=['HH_NET_INCOME_MONTHLY'], axis='rows', inplace=True)
    # Convert to yearly income
    df.HH_NET_INCOME_MONTHLY = df.HH_NET_INCOME_MONTHLY * 12
    # Rename column
    df = df.rename(columns={'HH_NET_INCOME_MONTHLY': 'HH_NET_INCOME_YEAR'})

    df.to_pickle(ppj('OUT_DATA', 'hh_inc.pkl'))


if __name__ == '__main__':
    main()
