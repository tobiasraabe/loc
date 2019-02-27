import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj


VARIABLE_NAMES_PEQUIV = {
    # 'cid': 'ID_ORIGINAL_HH',  # Case id, id of original household
    "syear": "YEAR",  # survey year
    # 'hid': 'ID_HH',  # current household id
    "pid": "ID",  # permanent personal id
    "d11109": "YEARS_EDUCATION",
}

RETAINED_COLUMNS_PEQUIV = list(VARIABLE_NAMES_PEQUIV.keys())


def main():
    df = pd.read_stata(ppj("IN_DATA", "pequiv.dta"), columns=RETAINED_COLUMNS_PEQUIV)
    df = df.rename(columns=VARIABLE_NAMES_PEQUIV)

    df.loc[df.YEARS_EDUCATION < 0, "YEARS_EDUCATION"] = np.nan

    df.YEARS_EDUCATION = df.groupby("ID").YEARS_EDUCATION.transform(
        lambda x: x.fillna(method="ffill")
    )

    df.to_pickle(ppj("OUT_DATA", "edu_years.pkl"))


if __name__ == "__main__":
    main()
