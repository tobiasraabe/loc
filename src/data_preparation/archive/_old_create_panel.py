"""This module creates the panel data set from the SOEP data files."""
from collections import OrderedDict

import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join as ppj

from src import EVENT_VARIABLES


NAN_IDENTIFIERS = [
    "[-1] keine Angabe",
    "[-2] trifft nicht zu",
    "[-3] nicht valide",
    "[-5] In Fragebogenversion nicht enthalten",
    "[-8] Frage in diesem Jahr nicht Teil des Frageprograms",
]

MONTH_DICT = OrderedDict(
    [
        ("[1] Januar", "January"),
        ("[2] Februar", "February"),
        ("[3] Maerz", "March"),
        ("[4] April", "April"),
        ("[5] Mai", "May"),
        ("[6] Juni", "June"),
        ("[7] Juli", "July"),
        ("[8] August", "August"),
        ("[9] September", "September"),
        ("[10] Oktober", "October"),
        ("[11] November", "November"),
        ("[12] Dezember", "December"),
    ]
)

VARIABLE_DICT_PL = {
    # General variables
    "cid": "ID_ORIGINAL_HH",  # Case id, id of original household
    "syear": "YEAR",  # survey year
    "hid": "ID_HH",  # current household id
    "pid": "ID",  # permanent personal id
    # Characteristics
    "pla0009": "GENDER",  # Gender
    "plb0022": "EMPLOYMENT_STATUS",  # Employment status
    "plb0304": "REASON_JOB_TERMINATED",  # Why job terminated
    "pld0131": "MARITAL_STATUS",  # Marital status
    # 'pld0135': 'MARRIED_MONTH_SY',  # Month married survey year
    # 'pld0136': 'MARRIED_MONTH_PY',  # Month married previous year
    "pld0141": "DIVORCED_MONTH_SY",  # Month divorced survey year, doc error
    "pld0142": "DIVORCED_MONTH_PY",  # Month divorced previous year
    "pld0144": "SEPARATED_MONTH_SY",  # Month separated survey year
    "pld0145": "SEPARATED_MONTH_PY",  # Month separated previous year
    "pld0147": "DEATH_PARTNER_MONTH_SY",  # partner died, survey year
    "pld0148": "DEATH_PARTNER_MONTH_PY",  # partner died, previous year
    "pld0156": "HH_COMP_CHANGE_MONTH_PY",  # change of household comp, py
    "pld0157": "HH_COMP_CHANGE_MONTH_SY",  # change of household comp, sy
    "pld0161": "DEATH_FATHER_MONTH_SY",  # father died survey year
    "pld0162": "DEATH_FATHER_MONTH_PY",  # father died previous year
    "pld0164": "DEATH_MOTHER_MONTH_SY",  # mother died survey year
    "pld0165": "DEATH_MOTHER_MONTH_PY",  # mother died previous year
    "pld0167": "DEATH_CHILD_MONTH_SY",  # child died survey year
    "pld0168": "DEATH_CHILD_MONTH_PY",  # child died previous year
    "pld0170": "DEATH_HH_PERSON_MONTH_SY",  # person in household died, sy
    "pld0171": "DEATH_HH_PERSON_MONTH_PY",  # person in household died, py
    "ple0010": "BIRTH_YEAR",  # birth year
    "ple0041": "LEGALLY_HANDICAPPED_PERC",  # legally handicapped, reduced
    # employment
    # LoC items
    "plh0005": "LOC_INFLUENCE_SOCIAL_COND",  # influence on social conditions
    # through involvement
    "plh0128": "LOC_ACHIEVED_DESERVE",  # have not achieved what I deserve
    "plh0245": "LOC_LIFES_COURSE",  # my lifes course depends on me
    "plh0246": "LOC_LUCK",  # what you achieve depends on luck
    "plh0247": "LOC_OTHERS",  # others make crucial decisions in my life
    "plh0248": "LOC_SUCCESS",  # success takes hard work
    "plh0249": "LOC_DOUBT",  # doubt my abilities when problems arise
    "plh0250": "LOC_POSSIBILITIES",  # possibilities are defined by social
    # conditions
    "plh0251": "LOC_ABILITIES",  # abilities are more important than effort
    "plh0252": "LOC_LITTLE_CONTROL",  # little control over my life
}

RETAINED_COLUMNS_PL = list(VARIABLE_DICT_PL.keys())

VARIABLE_DICT_PGEN = {"pid": "ID", "syear": "YEAR", "pgmonth": "INT_MONTH"}

RETAINED_COLUMNS_PGEN = list(VARIABLE_DICT_PGEN.keys())


def fill_with_mode(x):
    try:
        return x.value_counts().index[0]
    except IndexError:
        return np.nan


def clean_categoricals_from_multiple_nans(df, nan_list):
    """Cleans categoricals by replacing multiple NaN statements with np.nan and
    then removes the missing categories from the categorical index.
    """
    # Replace different NaN statements with np.nan
    df.replace(to_replace=NAN_IDENTIFIERS, value=np.nan, inplace=True)
    # Remove unused categories in categoricals
    categorical_names = list(df.select_dtypes("category").columns)
    for cat in categorical_names:
        df[cat].cat.remove_unused_categories(inplace=True)

    return df


def reorder_month_categoricals(df):
    month_variables = [i for i in df if "_MONTH" in i]

    for variable in month_variables:
        df[variable].cat.set_categories(MONTH_DICT.keys(), ordered=True, inplace=True)
        df[variable].cat.rename_categories(MONTH_DICT, inplace=True)

    return df


def create_panel():
    # Load dataset pl.dta
    df = pd.read_stata(ppj("IN_DATA", "pl.dta"), columns=RETAINED_COLUMNS_PL)
    # Rename columns
    df = df.rename(columns=VARIABLE_DICT_PL)
    # Sort df
    df.sort_values(["ID", "YEAR"], axis="rows", inplace=True)

    # Load dataset pgen.dta
    pgen = pd.read_stata(ppj("IN_DATA", "pgen.dta"), columns=RETAINED_COLUMNS_PGEN)
    # Rename columns
    pgen = pgen.rename(columns=VARIABLE_DICT_PGEN)
    # Merge with df
    df = df.merge(pgen, on=["ID", "YEAR"], how="left")

    # Post-merging procesing
    # Clean categoricals
    df = clean_categoricals_from_multiple_nans(df, NAN_IDENTIFIERS)
    # Relabel data containing months and make them comparable
    df = reorder_month_categoricals(df)

    # Dropping observations which have NaNs in their loc elicitations in 2005,
    # 2010 or 2015
    df = df.loc[
        (
            df[[i for i in df if "LOC" in i]].isnull().any(axis=1)
            & df.YEAR.isin([2005, 2010, 2015])
        )
        == 0
    ]

    # Shift LEGALLY_HANDICAPPED_PERC to the next year to be able to compute the
    # annual change in disability
    df["LEGALLY_HANDICAPPED_PERC_SHIFTED"] = df.groupby(
        "ID"
    ).LEGALLY_HANDICAPPED_PERC.transform("shift")

    # Select only observations which are in one of the two ranges, 2005-2010 or
    # 2010-2015, and which are complete, meaning having 6 observations for one
    # range or 11 for two.
    # First, create variables to indicate complete ranges
    df["YEAR_2005_2010"] = df.YEAR.isin([2005, 2006, 2007, 2008, 2009, 2010])
    df["YEAR_2005_2010_SUM"] = df.groupby("ID").YEAR_2005_2010.transform(sum)
    df["YEAR_2010_2015"] = df.YEAR.isin([2010, 2011, 2012, 2013, 2014, 2015])
    df["YEAR_2010_2015_SUM"] = df.groupby("ID").YEAR_2010_2015.transform(sum)
    # Select only valid years when range is complete
    df = df.loc[
        (
            df.YEAR.isin([2005, 2006, 2007, 2008, 2009, 2010])
            & (df.YEAR_2005_2010_SUM == 6)
        )
        | (
            df.YEAR.isin([2010, 2011, 2012, 2013, 2014, 2015])
            & (df.YEAR_2010_2015_SUM == 6)
        )
    ]
    # Test that for each individual, there are only 6 or 11 possible
    # observations. There are 6 if individuals are only observed over one
    # period, 2005-2010 or 2010-2015, and 11 if they are observed over the
    # whole range.
    assert df.groupby("ID").YEAR.count().isin([6, 11]).all()
    # Drop temporary columns
    df.drop(
        [
            "YEAR_2005_2010",
            "YEAR_2005_2010_SUM",
            "YEAR_2010_2015",
            "YEAR_2010_2015_SUM",
        ],
        axis="columns",
        inplace=True,
    )

    return df


def extract_loc(df):
    # Copy loc dataframe and drop columns in other frame
    loc = df.loc[
        df.YEAR.isin([2005, 2010, 2015]), ["ID", "YEAR"] + [i for i in df if "LOC" in i]
    ].copy()
    df.drop([i for i in df if "LOC" in i], axis="columns", inplace=True)

    # Save loc
    loc.to_pickle(ppj("OUT_DATA", "loc_raw.pkl"))

    return df


def clean_common_variables(df):
    # BIRTH_YEAR
    # Replace -5 with np.nan
    df.BIRTH_YEAR.replace(to_replace=-5, value=np.nan, inplace=True)
    filled_birth_year = df.groupby("ID").BIRTH_YEAR.transform(fill_with_mode)
    df.BIRTH_YEAR = filled_birth_year
    # Create age variable
    df["AGE"] = df.YEAR - df.BIRTH_YEAR
    # Drop BIRTH_YEAR
    df.drop("BIRTH_YEAR", axis="columns", inplace=True)

    # EMPLOYMENT_STATUS
    employment_dict = {
        "[1] Voll erwerbstaetig": "Full-Time Employment",
        "[2] Teilzeitbeschaeftigung": "Regular Part-Time Employment",
        "[3] Ausbildung,Lehre": "Vocational Training",
        "[4] Geringfuegig beschaeftigt": "Marginally employed",
        "[5] Altersteilzeit mit Arbeitszeit Null": "Near Retirement, Zero "
        "Working Hours",
        "[6] Freiwilliger Wehrdienst": "Voluntary Military Service",
        "[7] Freiwsoziales/oekol.Jahr, Bundesfreiwilligendienst": "Vol. Soc. "
        "Y. / Vol. Eco. Y. / Feder. Vol. Srvc",
        "[8] Werkstatt fuer behinderte Menschen": "Sheltered workshop",
        "[9] Nicht erwerbstaetig": "Not Employed",
    }
    df.EMPLOYMENT_STATUS.cat.rename_categories(employment_dict, inplace=True)
    df.EMPLOYMENT_STATUS.cat.as_unordered(inplace=True)
    # Drop three missing values in EMPLOYMENT_STATUS
    # df = df.loc[df.EMPLOYMENT_STATUS.notnull()]

    # GENDER
    # Rename categories
    gender_dict = {"[1] Maennlich": "Male", "[2] Weiblich": "Female"}
    df.GENDER.cat.rename_categories(gender_dict, inplace=True)
    # Fill NaNs in GENDER with the most common value per ID
    filled_gender = df.groupby("ID").GENDER.transform(fill_with_mode)
    df.GENDER = filled_gender.astype("category")

    # LEGALLY_HANDICAPPED_PERC
    # Replace negative values with NaNs
    legally_handicapped_perc_dict = {
        -1: 0,  # no answer/don't know
        -2: 0,  # does not apply
        -3: 70,  # implausible value, only applies to ID == 2722302 who has 70%
        # in the previous year. Assume it is constant.
    }
    df.replace(
        {
            "LEGALLY_HANDICAPPED_PERC": legally_handicapped_perc_dict,
            "LEGALLY_HANDICAPPED_PERC_SHIFTED": legally_handicapped_perc_dict,
        },
        inplace=True,
    )
    # Calculate change in LEGALLY_HANDICAPPED_PERC
    df["LEGALLY_HANDICAPPED_PERC_CHANGE"] = (
        df.LEGALLY_HANDICAPPED_PERC - df.LEGALLY_HANDICAPPED_PERC_SHIFTED
    )
    # Drop unneccessary columns
    df.drop(
        ["LEGALLY_HANDICAPPED_PERC", "LEGALLY_HANDICAPPED_PERC_SHIFTED"],
        axis="columns",
        inplace=True,
    )

    # MARITAL STATUS
    marital_status_dict = {
        "[1] Verheiratet, zusammenlebend": "Married, live together",
        "[2] Verheiratet, getrenntlebend": "Married, separated",
        "[3] Ledig": "Single",
        "[4] Geschieden, eing. gleichg. Partn. aufgehoben": "Divorced, "
        "registered partnership dissolved",
        "[5] Verwitwet, Lebenspartner/in verstorben": "Widowed",
        "[6] Eing. gleichg. Partn., zusammenlebend": "Registered same-sex "
        "partnership, living together",
        "[7] Eing. gleichg. Partn., getrenntlebend": "Registered same-sex "
        "partnership, separated",
    }
    df.MARITAL_STATUS.cat.rename_categories(marital_status_dict, inplace=True)

    # As Preuss, Hennecke (2017), we only consider plant closure and
    # displacement by employer to be sufficiently exogenous. Other reasons are
    # discarded.
    # Rename useful categories
    reason_job_terminated_dict = {
        "[1] Betriebsstillegung, Aufloesung Dienstst.": "Plant closure",
        "[3] Kuendigung Arbeitgeber": "Displacement by employer",
    }
    df.REASON_JOB_TERMINATED.cat.rename_categories(
        reason_job_terminated_dict, inplace=True
    )
    # Delete useless categories
    reason_job_terminated_list = [
        "[2] Eigene Kuendigung",
        "[4] Aufloesungsvertrag",
        "[5] Ende Befristung",
        "[6] Verrentung, Pensionierung",
        "[7] Beurlaubung",
        "[8] Geschaeftsaufgabe (Selbstaendige)",
        "[9] Vorruhestand",
        "[10] Ende der Ausbildung",
        "[11] Versetzung auf eigenen Wunsch",
        "[12] Versetzung durch Betrieb",
        "[13] Sonstige Gruende",
    ]
    df.REASON_JOB_TERMINATED.cat.remove_categories(
        reason_job_terminated_list, inplace=True
    )

    return df


def clean_event_variables(df):
    for var in EVENT_VARIABLES:
        # Shift var_MONTH_PY in the previous year
        df[var + "_MONTH_PY_SHIFTED"] = df.groupby("ID")[var + "_MONTH_PY"].shift(-1)
        # Use var_MONTH_SY where both agree
        df.loc[
            df[var + "_MONTH_SY"] == df[var + "_MONTH_PY_SHIFTED"], var + "_MONTH"
        ] = df[var + "_MONTH_SY"]
        # Use var_MONTH_PY_SHIFTED where var_MONTH_SY is NaN
        df.loc[
            df[var + "_MONTH_SY"].isnull() & df[var + "_MONTH_PY_SHIFTED"].notnull(),
            var + "_MONTH",
        ] = df[var + "_MONTH_PY_SHIFTED"]
        # Use var_MONTH_SY where var_MONTH_PY_SHIFTED is NaN or they
        # disagree because the PY could be more error prone to memory loss
        df.loc[
            df[var + "_MONTH_SY"].notnull() & df[var + "_MONTH_PY_SHIFTED"].isnull(),
            var + "_MONTH",
        ] = df[var + "_MONTH_SY"]
        df.loc[
            df[var + "_MONTH_SY"] != df[var + "_MONTH_PY_SHIFTED"], var + "_MONTH"
        ] = df[var + "_MONTH_SY"]
        # Make var_MONTH a categorical
        df[var + "_MONTH"] = df[var + "_MONTH"].astype("category")
        df[var + "_MONTH"].cat.set_categories(
            MONTH_DICT.values(), ordered=True, inplace=True
        )

        # Create variable whether var was before the interview to determine
        # timing. Note that, cases where the months of the event and interview
        # coincide are flagged as False.
        df.loc[df[var + "_MONTH"].notnull(), var + "_BEFORE_INTERVIEW"] = (
            df[var + "_MONTH"] < df.INT_MONTH
        )
        # There are some cases in which interview and var coincide. An event
        # happened before the interview if var_MONTH_SY is not NaN. The
        # opposite case if var_MONTH_SY is NaN is already flagged as false due
        # to the previous step
        df.loc[
            (df[var + "_MONTH"] == df.INT_MONTH) & df[var + "_MONTH_SY"].notnull(),
            var + "_BEFORE_INTERVIEW",
        ] = True

    # Separate the sample in the two periods, 2005-2010 and 2010-2015.
    df_2005_2010 = df.loc[df.YEAR.between(2005, 2010)].copy()
    df_2010_2015 = df.loc[df.YEAR.between(2010, 2015)].copy()

    # Delete events which are not occurring in the specific periods
    for var in EVENT_VARIABLES:
        # Delete events before interview in 2005
        df_2005_2010.loc[
            (df_2005_2010.YEAR == 2005) & df_2005_2010[var + "_BEFORE_INTERVIEW"],
            var + "_MONTH",
        ] = np.nan
        # Delete events in the first period after the interview
        df_2005_2010.loc[
            (df_2005_2010.YEAR == 2010)
            & (df_2005_2010[var + "_BEFORE_INTERVIEW"] == 0),
            var + "_MONTH",
        ] = np.nan
        # Delete events in the second period before the interview
        df_2010_2015.loc[
            (df_2010_2015.YEAR == 2010) & df_2010_2015[var + "_BEFORE_INTERVIEW"],
            var + "_MONTH",
        ] = np.nan
        # Delete events after interview in 2015
        df_2010_2015.loc[
            (df_2010_2015.YEAR == 2015)
            & (df_2010_2015[var + "_BEFORE_INTERVIEW"] == 0),
            var + "_MONTH",
        ] = np.nan

    # Create event identifiers, ongoing counts of current events and ongoing
    # counts of previous events.
    for df in [df_2005_2010, df_2010_2015]:
        for var in EVENT_VARIABLES:
            # Create event identifier
            df.loc[df[var + "_MONTH"].notnull(), "EVENT_" + var] = True
            # Create ongoing count of events per period
            # astype(float) to convert True to 1.0, cumsum to count, ffill to
            # to overwrite NaNs in following columns with previous values,
            # fillna(0) to convert NaNs at the beginning to zeros.
            df["EVENT_" + var + "_COUNT"] = df.groupby("ID")["EVENT_" + var].transform(
                lambda x: x.astype(float).cumsum().fillna(method="ffill").fillna(0)
            )
            # Create ongoing count of previous events per period
            df["EVENT_" + var + "_COUNT_PREVIOUS"] = (
                df["EVENT_" + var + "_COUNT"] - 1
            ).clip(lower=0)

    return df_2005_2010, df_2010_2015


def create_valid_event_variables_and_covariates(unaltered_df, event):
    # Create a copy of the dataframe to not alter the passed df
    df = unaltered_df.copy()
    # Create list of covariates
    covariates = [i for i in EVENT_VARIABLES if i != event]

    for cov in covariates:
        # Create identifier whether the specific event happened before the
        # event of a covariate. Important is that there are only 5 cases
        # where events happened in the same month. Therefore, we will treat
        # them as if they have occurred simultaneously.
        df[event + "_BEFORE_" + cov] = df[event + "_MONTH"] < df[cov + "_MONTH"]
        # Change ongoing event count of covariate by -1 if event happened
        # before the covariate event.
        cond_same_year_event = (
            df[cov + "_MONTH"].notnull() & df[event + "_MONTH"].notnull()
        )
        df.loc[
            df[event + "_BEFORE_" + cov] & cond_same_year_event,
            "EVENT_" + cov + "_COUNT",
        ] = (df["EVENT_" + cov + "_COUNT"] - 1)

    return df


def drop_unused_columns_and_observations(df, event_name):
    # Drop columns
    unused_columns = []
    unused_columns += [i for i in df if "SY" in i]
    unused_columns += [i for i in df if "PY" in i]
    unused_columns += [i for i in df if "_MONTH" in i]
    unused_columns += [i for i in df if "BEFORE_INTERVIEW" in i]
    unused_columns += [i for i in df if event_name + "_BEFORE_" in i]
    unused_columns += [i for i in df if ("COUNT_PREVIOUS" in i) & ~(event_name in i)]
    unused_columns += ["EVENT_" + i for i in EVENT_VARIABLES if i != event_name]
    df.drop(unused_columns, axis="columns", inplace=True)
    # Drop observations
    # df = df.loc[df['EVENT_' + event_name]]

    return df


if __name__ == "__main__":
    # Create a unaggregated panel from 2005 to 2015
    df = create_panel()
    # Extract LOC for separate processing
    df = extract_loc(df)
    # Clean common variables
    df = clean_common_variables(df)
    # Clean event variables
    df_2005_2010, df_2010_2015 = clean_event_variables(df)
    # For each event, process both periods by creating
    #       1. event identifier and counts of the specific event
    #       2. event counts of other events in relation to each specific event
    #          to have valid counts of other events at the specific event as
    #          covariates.
    #       3. Reduce panel.
    for event in EVENT_VARIABLES:
        container = []
        for df in [df_2005_2010, df_2010_2015]:
            df_temp = create_valid_event_variables_and_covariates(df, event)
            container.append(df_temp)
        # Merge both periods for one event to one dataframe
        df_temp = container[0]
        df_merged = df_temp.append(container[1])
        # Clean event data
        df_merged = drop_unused_columns_and_observations(df_merged, event)
        # Save the current dataframe
        df_merged.to_pickle(ppj("OUT_DATA", f"panel_{event.lower()}.pkl"))
