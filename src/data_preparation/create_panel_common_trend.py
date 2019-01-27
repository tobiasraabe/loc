import pandas as pd
from bld.project_paths import project_paths_join as ppj


EVENT_VARIABLES = [
    "CHILD_DISORDER",
    "DEATH_CHILD",
    "DEATH_FATHER",
    "DEATH_HH_PERSON",
    "DEATH_MOTHER",
    "DEATH_PARTNER",
    "DIVORCED",
    "HH_COMP_CHANGE",
    "LAST_JOB_ENDED_LIMITED",
    "LEGALLY_HANDICAPPED_PERC",
    "PREGNANCY_UNPLANNED",
    "SEPARATED",
]


def restrict_sample(df):
    """This function restricts the sample in the following way:
           1. Only observations observed from 2005-2015
           2. Only observations without events in 2005-2010
           3. Only the period 2010-2015
    """
    # 1. Restriction
    set_id_2010 = set(df.loc[df.YEAR == 2010, "ID"].unique())
    set_id_2015 = set(df.loc[df.YEAR == 2015, "ID"].unique())
    # Intersection of IDs
    set_id = set_id_2010 & set_id_2015
    # Select valid observations
    df = df.loc[df.ID.isin(set_id)].copy()

    # 2. and 3. Restriction
    df["EVENT_ALL_COUNT"] = df[["EVENT_" + i + "_COUNT" for i in EVENT_VARIABLES]].sum(
        axis=1
    )
    valid_ids = df.loc[(df.YEAR == 2010) & (df.EVENT_ALL_COUNT == 0), "ID"].values
    df = df.loc[df.ID.isin(valid_ids)].copy()

    # Drop unused variables
    df.drop("EVENT_ALL_COUNT", axis="columns", inplace=True)

    return df


def create_leads(df):
    # Split the sample
    df_2010 = df.loc[df.YEAR == 2010].copy()
    df_2015 = df.loc[df.YEAR == 2015].copy()
    # Extract ids and event identifier from the 2015 period
    df_leads = df_2015[["ID"] + ["EVENT_" + i for i in EVENT_VARIABLES]].copy()
    # Merge onto previous data to get leads
    df_2010 = df_2010.merge(df_leads, on="ID", how="left", suffixes=("", "_LEADS"))
    # Merge both periods
    df = df_2010.append(df_2015)

    # Fill NaNs due to merge with False
    df[[i for i in df if "LEADS" in i]] = df.loc[
        :, [i for i in df if "_LEADS" in i]
    ].fillna(False)

    return df


def prepare_for_regression(df):
    # Merge with loc
    loc = pd.read_pickle(ppj("OUT_DATA", "loc_fa.pkl"))
    # Merge FIRST_FACTOR_DELTA in panel
    df = df.merge(
        loc[["ID", "YEAR", "FIRST_FACTOR_DELTA"]], on=["ID", "YEAR"], how="left"
    )

    # Form 8 age groups. Each group encompasses about one decade.
    df["AGE_GROUPS"] = pd.cut(df["AGE"], list(range(20, 81, 10)) + [105])

    list_counts = [
        i
        for i in df
        if "EVENT" in i
        and "COUNT" in i
        and "PREVIOUS" not in i
        and "LIMITED" not in i
        and "LEADS" not in i
    ]
    # Set up column with counts of all traumatic events.
    df["EVENT_COUNTS_ALL"] = df[list_counts].sum(axis=1)

    # Replace all 0 values in net hh income with 2 as ln(0) is not possible in
    # regression.
    df["HH_NET_INCOME_YEAR"] = df["HH_NET_INCOME_YEAR"].replace([0.0], [2.0])

    return df


if __name__ == "__main__":
    # Load dataframe
    df = pd.read_pickle(ppj("OUT_DATA", "panel.pkl"))
    # Restrict sample
    df = restrict_sample(df)
    # Create leads
    df = create_leads(df)
    # Prepare for regression
    df = prepare_for_regression(df)
    # Save
    df.to_pickle(ppj("OUT_DATA", "panel_common_trend.pkl"))
