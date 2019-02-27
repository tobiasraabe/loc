"""Prepare ``pgen.dta`` to extract education groups.

"""
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj


NAN_IDENTIFIERS = [
    "[-1] keine Angabe",
    "[-2] trifft nicht zu",
    "[-3] nicht valide",
    "[-5] In Fragebogenversion nicht enthalten",
    "[-8] Frage in diesem Jahr nicht Teil des Frageprograms",
]

VARIABLE_NAMES_PGEN = {
    # 'cid': 'ID_ORIGINAL_HH',  # Case id, id of original household
    "syear": "YEAR",  # survey year
    # 'hid': 'ID_HH',  # current household id
    "pid": "ID",  # permanent personal id
    "pgcasmin": "EDUCATION_GROUPS_CASMIN",
    "pgisced97": "EDUCATION_GROUPS_ISCED97",
    "pgisced11": "EDUCATION_GROUPS_ISCED11",
}

RETAINED_COLUMNS_PGEN = list(VARIABLE_NAMES_PGEN.keys())


def main():
    df = pd.read_stata(ppj("IN_DATA", "pgen.dta"), columns=RETAINED_COLUMNS_PGEN)
    df = df.rename(columns=VARIABLE_NAMES_PGEN)

    df.replace(to_replace=NAN_IDENTIFIERS, value=np.nan, inplace=True)
    for i in [i for i in df if "EDUCATION" in i]:
        df[i].cat.remove_unused_categories(inplace=True)

    # Fill NaNs conservatively meaning only a forward fill since educational
    # qualification cannot be lost. (In a previous version, we also backfilled values.)
    for var in [i for i in df if "EDUCATION" in i]:
        df[var] = df.groupby("ID")[var].transform(lambda x: x.fillna(method="ffill"))

    df.to_pickle(ppj("OUT_DATA", "edu_groups.pkl"))


if __name__ == "__main__":
    main()
