#!/usr/bin/env python

"""This module creates the panel data set from the SOEP data files."""


import calendar
import datetime
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from collections import OrderedDict
from dateutil.relativedelta import relativedelta


EVENT_VARIABLES = ['CHILD_DISORDER', 'DEATH_CHILD', 'DEATH_FATHER',
                   'DEATH_HH_PERSON', 'DEATH_MOTHER', 'DEATH_PARTNER',
                   'DIVORCED', 'HH_COMP_CHANGE', 'LAST_JOB_ENDED',
                   'PREGNANCY_UNPLANNED', 'SEPARATED']

NAN_IDENTIFIERS = [
    '[-1] keine Angabe', '[-2] trifft nicht zu',
    '[-3] nicht valide',
    '[-5] In Fragebogenversion nicht enthalten',
    '[-8] Frage in diesem Jahr nicht Teil des Frageprograms',
]

MONTH_DICT = OrderedDict([('[1] Januar', 'January'),
                          ('[2] Februar', 'February'),
                          ('[3] Maerz', 'March'),
                          ('[4] April', 'April'),
                          ('[5] Mai', 'May'),
                          ('[6] Juni', 'June'),
                          ('[7] Juli', 'July'),
                          ('[8] August', 'August'),
                          ('[9] September', 'September'),
                          ('[10] Oktober', 'October'),
                          ('[11] November', 'November'),
                          ('[12] Dezember', 'December')])

VARIABLE_DICT_PL = {
    # General variables
    'cid': 'ID_ORIGINAL_HH',  # Case id, id of original household
    'syear': 'YEAR',  # survey year
    'hid': 'ID_HH',  # current household id
    'pid': 'ID',  # permanent personal id
    # Characteristics
    'pla0009': 'GENDER',  # Gender
    'plb0022': 'EMPLOYMENT_STATUS',  # Employment status, potentially sy
                                     # because of comparison with
                                     # LAST_JOB_ENDED_MONTH
    'pld0131': 'MARITAL_STATUS',  # Marital status
    'plb0298': 'LAST_JOB_ENDED_MONTH_PY',  # last job ended, previous year
    'plb0299': 'LAST_JOB_ENDED_MONTH_SY',  # last job ended, survey year
    'plb0304': 'REASON_JOB_TERMINATED',  # Why job terminated
    # 'pld0135': 'MARRIED_MONTH_SY',  # Month married survey year
    # 'pld0136': 'MARRIED_MONTH_PY',  # Month married previous year
    'pld0141': 'DIVORCED_MONTH_SY',  # Month divorced survey year, doc error
    'pld0142': 'DIVORCED_MONTH_PY',  # Month divorced previous year
    'pld0144': 'SEPARATED_MONTH_SY',  # Month separated survey year
    'pld0145': 'SEPARATED_MONTH_PY',  # Month separated previous year
    'pld0147': 'DEATH_PARTNER_MONTH_SY',  # partner died, survey year
    'pld0148': 'DEATH_PARTNER_MONTH_PY',  # partner died, previous year
    'pld0156': 'HH_COMP_CHANGE_MONTH_PY',  # change of household comp, py
    'pld0157': 'HH_COMP_CHANGE_MONTH_SY',  # change of household comp, sy
    'pld0161': 'DEATH_FATHER_MONTH_SY',  # father died survey year
    'pld0162': 'DEATH_FATHER_MONTH_PY',  # father died previous year
    'pld0164': 'DEATH_MOTHER_MONTH_SY',  # mother died survey year
    'pld0165': 'DEATH_MOTHER_MONTH_PY',  # mother died previous year
    'pld0167': 'DEATH_CHILD_MONTH_SY',  # child died survey year
    'pld0168': 'DEATH_CHILD_MONTH_PY',  # child died previous year
    'pld0170': 'DEATH_HH_PERSON_MONTH_SY',  # person in household died, sy
    'pld0171': 'DEATH_HH_PERSON_MONTH_PY',  # person in household died, py
    'ple0010': 'BIRTH_YEAR',  # birth year
    'ple0041': 'LEGALLY_HANDICAPPED_PERC',  # legally handicapped, reduced
                                            # employment
    # LoC items
    'plh0005': 'LOC_INFLUENCE_SOCIAL_COND',  # influence on social conditions
                                             # through involvement
    'plh0128': 'LOC_ACHIEVED_DESERVE',  # have not achieved what I deserve
    'plh0245': 'LOC_LIFES_COURSE',  # my lifes course depends on me
    'plh0246': 'LOC_LUCK',  # what you achieve depends on luck
    'plh0247': 'LOC_OTHERS',  # others make crucial decisions in my life
    'plh0248': 'LOC_SUCCESS',  # success takes hard work
    'plh0249': 'LOC_DOUBT',  # doubt my abilities when problems arise
    'plh0250': 'LOC_POSSIBILITIES',  # possibilities are defined by social
                                     # conditions
    'plh0251': 'LOC_ABILITIES',  # abilities are more important than effort
    'plh0252': 'LOC_LITTLE_CONTROL',  # little control over my life
}

RETAINED_COLUMNS_PL = list(VARIABLE_DICT_PL.keys())

VARIABLE_DICT_PGEN = {
    'pid': 'ID',
    'syear': 'YEAR',
    'pgmonth': 'INT_MONTH'
}

RETAINED_COLUMNS_PGEN = list(VARIABLE_DICT_PGEN.keys())


MONTH_TO_NUMBER = {v: k for k, v in enumerate(calendar.month_name)}
MONTH_TO_NUMBER.pop('')


def calculate_time_difference_between_event_int(
        x, event, end_year_interview):
    try:
        x[event + '_MONTH']
        start_month = x[event + '_MONTH']
        start_year = x.YEAR
        end_month = x[f'INT_MONTH_{end_year_interview}']
        end_year = end_year_interview

        start_month_number = MONTH_TO_NUMBER[start_month]
        end_month_number = MONTH_TO_NUMBER[end_month]

        start = datetime.datetime.strptime(
            f'{start_year} {start_month_number}', '%Y %m')
        end = datetime.datetime.strptime(
            f'{end_year} {end_month_number}', '%Y %m')

        time_diff = relativedelta(end, start)

        difference_in_months = time_diff.years * 12 + time_diff.months

        return difference_in_months
    except KeyError:
        return np.nan


def assert_same_number_of_observations(func):
    """This decorator ensures that the number of observations does not change
    by this transformation of the dataframe."""
    def wrapper(*args, **kwargs):
        num_obs_before_transformation = df.shape[0]
        transformed_df = func(*args, **kwargs)

        num_obs_after_transformation = transformed_df.shape[0]
        assert num_obs_before_transformation == num_obs_after_transformation

        return transformed_df
    return wrapper


def report_shape_of_dataframe(func):
    def wrapper(*args, **kwargs):
        transformed_df = func(*args, **kwargs)

        shape = transformed_df.shape
        print(f'Shape is now: {shape}.')

        return transformed_df
    return wrapper


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
    categorical_names = list(df.select_dtypes('category').columns)
    for cat in categorical_names:
        df[cat].cat.remove_unused_categories(inplace=True)

    return df


def reorder_month_categoricals(df):
    month_variables = [i for i in df if '_MONTH' in i]

    for variable in month_variables:
        df[variable].cat.set_categories(MONTH_DICT.keys(), ordered=True,
                                        inplace=True)
        df[variable].cat.rename_categories(MONTH_DICT, inplace=True)

    return df


def create_panel():
    # Load dataset pl.dta
    df = pd.read_stata(ppj('IN_DATA', 'pl.dta'), columns=RETAINED_COLUMNS_PL)
    # Rename columns
    df = df.rename(columns=VARIABLE_DICT_PL)
    # Sort df
    df.sort_values(['ID', 'YEAR'], axis='rows', inplace=True)

    # Load dataset pgen.dta
    pgen = pd.read_stata(ppj('IN_DATA', 'pgen.dta'),
                         columns=RETAINED_COLUMNS_PGEN)
    # Rename columns
    pgen = pgen.rename(columns=VARIABLE_DICT_PGEN)
    # Merge with df
    df = df.merge(pgen, on=['ID', 'YEAR'], how='left')

    # Post-merging procesing
    # Clean categoricals
    df = clean_categoricals_from_multiple_nans(df, NAN_IDENTIFIERS)
    # Relabel data containing months and make them comparable
    df = reorder_month_categoricals(df)

    # Dropping observations which have NaNs in their loc elicitations in 2005,
    # 2010 or 2015
    df = df.loc[
        (df[[i for i in df if 'LOC' in i]].isnull().any(axis=1) &
         df.YEAR.isin([2005, 2010, 2015])) == 0
    ]

    # Shift LEGALLY_HANDICAPPED_PERC to the next year to be able to compute the
    # annual change in disability
    df['LEGALLY_HANDICAPPED_PERC_SHIFTED'] = df.groupby(
        'ID').LEGALLY_HANDICAPPED_PERC.transform('shift')

    # Select only observations which are in one of the two ranges, 2005-2010 or
    # 2010-2015, and which are complete, meaning having 6 observations for one
    # range or 11 for two.
    # First, create variables to indicate complete ranges
    df['YEAR_2005_2010'] = df.YEAR.isin([2005, 2006, 2007, 2008, 2009, 2010])
    df['YEAR_2005_2010_SUM'] = df.groupby('ID').YEAR_2005_2010.transform(sum)
    df['YEAR_2010_2015'] = df.YEAR.isin([2010, 2011, 2012, 2013, 2014, 2015])
    df['YEAR_2010_2015_SUM'] = df.groupby('ID').YEAR_2010_2015.transform(sum)
    # Select only valid years when range is complete
    df = df.loc[(df.YEAR_2005_2010 & (df.YEAR_2005_2010_SUM == 6)) |
                (df.YEAR_2010_2015 & (df.YEAR_2010_2015_SUM == 6))]
    # Test that for each individual, there are only 6 or 11 possible
    # observations. There are 6 if individuals are only observed over one
    # period, 2005-2010 or 2010-2015, and 11 if they are observed over the
    # whole range.
    assert df.groupby('ID').YEAR.count().isin([6, 11]).all()

    return df


@report_shape_of_dataframe
@assert_same_number_of_observations
def clean_common_variables(df):
    # BIRTH_YEAR
    # Replace -5 with np.nan
    df.BIRTH_YEAR.replace(to_replace=-5, value=np.nan, inplace=True)
    filled_birth_year = df.groupby('ID').BIRTH_YEAR.transform(fill_with_mode)
    df.BIRTH_YEAR = filled_birth_year
    # Create age variable
    df['AGE'] = df.YEAR - df.BIRTH_YEAR
    # Drop BIRTH_YEAR
    df.drop('BIRTH_YEAR', axis='columns', inplace=True)

    # EMPLOYMENT_STATUS: Create states according to Preuss, Hennecke (2017) who
    # sort employed, part-time employed and self-employed to employed,
    # unemployed and others.
    employment_dict = {
        '[1] Voll erwerbstaetig': 'Employed',
        '[2] Teilzeitbeschaeftigung': 'Employed',
        '[3] Ausbildung,Lehre': 'Other',
        '[4] Geringfuegig beschaeftigt': 'Other',
        '[5] Altersteilzeit mit Arbeitszeit Null': 'Other',
        '[6] Freiwilliger Wehrdienst': 'Other',
        '[7] Freiwsoziales/oekol.Jahr, Bundesfreiwilligendienst': 'Other',
        '[8] Werkstatt fuer behinderte Menschen': 'Other',
        '[9] Nicht erwerbstaetig': 'Not Employed'
    }
    df.EMPLOYMENT_STATUS.replace(employment_dict, inplace=True)
    df.EMPLOYMENT_STATUS = df.EMPLOYMENT_STATUS.astype('category')

    # GENDER
    # Rename categories
    gender_dict = {'[1] Maennlich': 'Male', '[2] Weiblich': 'Female'}
    df.GENDER.cat.rename_categories(gender_dict, inplace=True)
    # Fill NaNs in GENDER with the most common value per ID
    filled_gender = df.groupby('ID').GENDER.transform(fill_with_mode)
    df.GENDER = filled_gender.astype('category')

    # LEGALLY_HANDICAPPED_PERC
    # Replace negative values with NaNs
    legally_handicapped_perc_dict = {
        -1: 0,   # no answer/don't know
        -2: 0,   # does not apply
        -3: 70,  # implausible value, only applies to ID == 2722302 who has 70%
                 # in the previous year. Assume it is constant.
    }
    df.replace({
        'LEGALLY_HANDICAPPED_PERC': legally_handicapped_perc_dict,
        'LEGALLY_HANDICAPPED_PERC_SHIFTED': legally_handicapped_perc_dict},
        inplace=True)
    # Calculate change in LEGALLY_HANDICAPPED_PERC
    df['LEGALLY_HANDICAPPED_PERC_CHANGE'] = (
        df.LEGALLY_HANDICAPPED_PERC - df.LEGALLY_HANDICAPPED_PERC_SHIFTED)
    # Drop unneccessary columns
    df.drop(['LEGALLY_HANDICAPPED_PERC', 'LEGALLY_HANDICAPPED_PERC_SHIFTED'],
            axis='columns', inplace=True)

    # MARITAL STATUS
    marital_status_dict = {
        '[1] Verheiratet, zusammenlebend': 'Relationship',
        '[2] Verheiratet, getrenntlebend': 'Single',
        '[3] Ledig': 'Single',
        '[4] Geschieden, eing. gleichg. Partn. aufgehoben': 'Single',
        '[5] Verwitwet, Lebenspartner/in verstorben': 'Single',
        '[6] Eing. gleichg. Partn., zusammenlebend': 'Relationship',
        '[7] Eing. gleichg. Partn., getrenntlebend': 'Single',
    }
    df.MARITAL_STATUS.replace(marital_status_dict, inplace=True)
    df.MARITAL_STATUS = df.MARITAL_STATUS.astype('category')

    # As Preuss, Hennecke (2017), we only consider plant closure and
    # displacement by employer to be sufficiently exogenous. Other reasons are
    # discarded.
    # Rename useful categories
    reason_job_terminated_dict = {
        '[1] Betriebsstillegung, Aufloesung Dienstst.': 'Plant closure',
        '[3] Kuendigung Arbeitgeber': 'Displacement by employer'}
    df.REASON_JOB_TERMINATED.cat.rename_categories(reason_job_terminated_dict,
                                                   inplace=True)
    # Delete useless categories
    reason_job_terminated_list = [
        '[2] Eigene Kuendigung', '[4] Aufloesungsvertrag',
        '[5] Ende Befristung', '[6] Verrentung, Pensionierung',
        '[7] Beurlaubung', '[8] Geschaeftsaufgabe (Selbstaendige)',
        '[9] Vorruhestand', '[10] Ende der Ausbildung',
        '[11] Versetzung auf eigenen Wunsch', '[12] Versetzung durch Betrieb',
        '[13] Sonstige Gruende']
    df.REASON_JOB_TERMINATED.cat.remove_categories(reason_job_terminated_list,
                                                   inplace=True)

    # Shift REASON_JOB_TERMINATED in the previous period because it is
    # ambiguous whether the value belongs to the survey year or previous year.
    df['REASON_JOB_TERMINATED_SHIFTED'] = df.groupby(
        'ID').REASON_JOB_TERMINATED.transform(lambda x: x.shift(-1))

    return df


@report_shape_of_dataframe
@assert_same_number_of_observations
def merge_with_edu_groups(df):
    # Load dataset
    edu_groups = pd.read_pickle(ppj('OUT_DATA', 'edu_groups.pkl'))
    # Merge with panel
    df = df.merge(edu_groups, on=['ID', 'YEAR'], how='left')

    return df


@report_shape_of_dataframe
@assert_same_number_of_observations
def merge_with_edu_years(df):
    # Load dataset
    edu_years = pd.read_pickle(ppj('OUT_DATA', 'edu_years.pkl'))
    # Merge with panel
    df = df.merge(edu_years, on=['ID', 'YEAR'], how='left')

    return df


@report_shape_of_dataframe
@assert_same_number_of_observations
def merge_with_migration(df):
    # Load dataset
    mig = pd.read_pickle(ppj('OUT_DATA', 'migration.pkl'))
    # Merge with panel
    df = df.merge(mig, on='ID', how='left')

    return df


@report_shape_of_dataframe
@assert_same_number_of_observations
def extract_loc(df):
    # Copy loc dataframe and drop columns in other frame
    loc = df.loc[df.YEAR.isin([2005, 2010, 2015]),
                 ['ID', 'YEAR', 'AGE'] + [i for i in df if 'LOC' in i]].copy()
    df.drop([i for i in df if 'LOC' in i], axis='columns', inplace=True)

    # Save loc
    loc.to_pickle(ppj('OUT_DATA', 'loc_raw.pkl'))

    return df


@report_shape_of_dataframe
@assert_same_number_of_observations
def merge_with_preg_dis(df):
    # Load other datasets
    preg = pd.read_pickle(ppj('OUT_DATA', 'preg.pkl'))
    dis = pd.read_pickle(ppj('OUT_DATA', 'dis.pkl'))
    # Merge with df
    df = df.merge(preg, how='left', on=['ID_HH', 'YEAR'])
    df = df.merge(dis, how='left', on=['ID_HH', 'YEAR'])
    # Combine multiple columns
    df['ID_MOTHER'] = df.ID_MOTHER_x.fillna(df.ID_MOTHER_y)
    df['MOTHER_PREGNANT_AT_PQ_YEAR'] = df.MOTHER_PREGNANT_AT_PQ_YEAR_x.fillna(
        df.MOTHER_PREGNANT_AT_PQ_YEAR_y)
    df.drop(['ID_MOTHER_x', 'ID_MOTHER_y', 'MOTHER_PREGNANT_AT_PQ_YEAR_x',
             'MOTHER_PREGNANT_AT_PQ_YEAR_y'], axis='columns', inplace=True)

    return df


@report_shape_of_dataframe
def clean_event_variables(df):

    for var in EVENT_VARIABLES:
        # These variables already have one column containing months and do not
        # need to be reduced.
        if var in ['CHILD_DISORDER', 'PREGNANCY_UNPLANNED']:
            pass
        else:
            # Shift var_MONTH_PY in the previous year
            df[var + '_MONTH_PY_SHIFTED'] = df.groupby(
                'ID')[var + '_MONTH_PY'].shift(-1)
            # Use var_MONTH_SY where both agree
            df.loc[df[var + '_MONTH_SY'] == df[var + '_MONTH_PY_SHIFTED'],
                   var + '_MONTH'] = df[var + '_MONTH_SY']
            # Use var_MONTH_PY_SHIFTED where var_MONTH_SY is NaN
            df.loc[df[var + '_MONTH_SY'].isnull() &
                   df[var + '_MONTH_PY_SHIFTED'].notnull(),
                   var + '_MONTH'] = df[var + '_MONTH_PY_SHIFTED']
            # Use var_MONTH_SY where var_MONTH_PY_SHIFTED is NaN or they
            # disagree because the PY could be more error prone to memory loss
            df.loc[df[var + '_MONTH_SY'].notnull() &
                   df[var + '_MONTH_PY_SHIFTED'].isnull(),
                   var + '_MONTH'] = df[var + '_MONTH_SY']
            df.loc[df[var + '_MONTH_SY'] != df[var + '_MONTH_PY_SHIFTED'],
                   var + '_MONTH'] = df[var + '_MONTH_SY']
            # Make var_MONTH a categorical
            df[var + '_MONTH'] = df[var + '_MONTH'].astype('category')
            df[var + '_MONTH'].cat.set_categories(MONTH_DICT.values(),
                                                  ordered=True, inplace=True)

        # Create variable whether var was before the interview to determine
        # timing. Note that, cases where the months of the event and interview
        # coincide are flagged as False.
        df.loc[df[var + '_MONTH'].notnull(),
               var + '_BEFORE_INTERVIEW'] = df[var + '_MONTH'] < df.INT_MONTH

        # There are some cases in which interview and var coincide. An event
        # happened before the interview if MOTHER_PREGNANT_AT_PQ_YEAR is the
        # same as the survey year.
        if var in ['CHILD_DISORDER', 'PREGNANCY_UNPLANNED']:
            df.loc[(df.INT_MONTH == df[var + '_MONTH']) &
                   df.MOTHER_PREGNANT_AT_PQ_YEAR.notnull() &
                   (df.YEAR == df.MOTHER_PREGNANT_AT_PQ_YEAR),
                   var + '_BEFORE_INTERVIEW'] = True
        # There are some cases in which interview and var coincide. An event
        # happened before the interview if var_MONTH_SY is not NaN. The
        # opposite case if var_MONTH_SY is NaN is already flagged as false due
        # to the previous step
        else:
            df.loc[(df[var + '_MONTH'] == df.INT_MONTH) &
                   df[var + '_MONTH_SY'].notnull(),
                   var + '_BEFORE_INTERVIEW'] = True

    # Save for exploration
    df.to_pickle(ppj('OUT_DATA', 'panel_inspection_2.pkl'))
    # Separate the sample in the two periods, 2005-2010 and 2010-2015. We
    # cannot simply use years from 2010-2015 for the second period, because
    # there would exist overhanging observations which have only a complete
    # history over the first period.
    df_2005_2010 = df.loc[df.YEAR_2005_2010 &
                          (df.YEAR_2005_2010_SUM == 6)].copy()
    df_2010_2015 = df.loc[df.YEAR_2010_2015 &
                          (df.YEAR_2010_2015_SUM == 6)].copy()

    # Delete events which are not occurring in the specific periods
    for var in EVENT_VARIABLES:
        # Delete events before interview in 2005
        df_2005_2010.loc[(df_2005_2010.YEAR == 2005) &
                         df_2005_2010[var + '_BEFORE_INTERVIEW'],
                         var + '_MONTH'] = np.nan
        # Delete events in the first period after the interview
        df_2005_2010.loc[(df_2005_2010.YEAR == 2010) &
                         (df_2005_2010[var + '_BEFORE_INTERVIEW'] == 0),
                         var + '_MONTH'] = np.nan
        # Delete events in the second period before the interview
        df_2010_2015.loc[(df_2010_2015.YEAR == 2010) &
                         df_2010_2015[var + '_BEFORE_INTERVIEW'],
                         var + '_MONTH'] = np.nan
        # Delete events after interview in 2015
        df_2010_2015.loc[(df_2010_2015.YEAR == 2015) &
                         (df_2010_2015[var + '_BEFORE_INTERVIEW'] == 0),
                         var + '_MONTH'] = np.nan

    # Preuss, Hennecke (2017) delete every observation with more than three
    # job losses between two LOC interviews. We need to save the current count
    # of displacements.
    df_2005_2010['LAST_JOB_ENDED_COUNT_FULL'] = df_2005_2010.groupby(
        'ID').LAST_JOB_ENDED_MONTH.transform('count')
    df_2010_2015['LAST_JOB_ENDED_COUNT_FULL'] = df_2010_2015.groupby(
        'ID').LAST_JOB_ENDED_MONTH.transform('count')
    # As PH (2017), we only focus on displacement by employer and plant
    # closure. Therefore, we will delete every other instances of job loss from
    # LAST_JOB_ENDED_MONTH. As all categories are eliminated from
    # REASON_JOB_TERMINATED which are not valid by the considerations of
    # Preuss, Hennecke (2017), we can eliminate all job losses where
    # REASON_JOB_TERMINATED and REASON_JOB_TERMINATED_SHIFTED are NaN.
    df_2005_2010.loc[df_2005_2010.REASON_JOB_TERMINATED.isnull() &
                     df_2005_2010.REASON_JOB_TERMINATED_SHIFTED.isnull(),
                     'LAST_JOB_ENDED_MONTH'] = np.nan
    df_2010_2015.loc[df_2010_2015.REASON_JOB_TERMINATED.isnull() &
                     df_2010_2015.REASON_JOB_TERMINATED_SHIFTED.isnull(),
                     'LAST_JOB_ENDED_MONTH'] = np.nan

    # Create event identifiers, ongoing counts of current events and ongoing
    # counts of previous events.
    for df in [df_2005_2010, df_2010_2015]:
        for var in EVENT_VARIABLES:
            # Create event identifier
            df.loc[df[var + '_MONTH'].notnull(), 'EVENT_' + var] = True
            # Create ongoing count of events per period
            # astype(float) to convert True to 1.0, cumsum to count, ffill to
            # to overwrite NaNs in following columns with previous values,
            # fillna(0) to convert NaNs at the beginning to zeros.
            df['EVENT_' + var + '_COUNT'] = df.groupby(
                'ID')['EVENT_' + var].transform(
                    lambda x: x.astype(float).cumsum().fillna(
                        method='ffill').fillna(0))
            # Create ongoing count of previous events per period
            df['EVENT_' + var + '_COUNT_PREVIOUS'] = (
                df['EVENT_' + var + '_COUNT'] - 1).clip(lower=0)

    # Now, we want to calculate the monthly difference of an event to the next
    # LOC interview in 2010 or 2015. First, we need to assign the month of the
    # next LOC interview to each observation by extracting the interview month
    # in 2010, 2015 and merging the series by IDs.
    int_month_2010 = df_2005_2010.loc[
        df_2005_2010.YEAR == 2010, ['ID', 'INT_MONTH']].copy()
    int_month_2015 = df_2010_2015.loc[
        df_2010_2015.YEAR == 2015, ['ID', 'INT_MONTH']].copy()

    df_2005_2010 = df_2005_2010.merge(int_month_2010, how='left', on='ID',
                                      suffixes=('', '_2010'))
    df_2010_2015 = df_2010_2015.merge(int_month_2015, how='left', on='ID',
                                      suffixes=('', '_2015'))
    # Calculate time difference for each event to the next interview month in
    # 2010 or 2015 and assign values to EVENT_var_TIME_DIFF
    for var in EVENT_VARIABLES:
        df_2005_2010['EVENT_' + var + '_TIME_DIFF'] = df_2005_2010.apply(
            calculate_time_difference_between_event_int, args=(var, 2010),
            axis=1)
        df_2010_2015['EVENT_' + var + '_TIME_DIFF'] = df_2010_2015.apply(
            calculate_time_difference_between_event_int, args=(var, 2015),
            axis=1)
        # At last, we propagate previous values to the next period if there is
        # a NaN. This means that the last value of EVENT_var_TIME_DIFF per ID
        # in 2010 or 2015 contains the time difference in months to the last
        # event. I do not know how to handle multiple durations. All other NaNs
        # will be filled with 0.
        df_2005_2010['EVENT_' + var + '_TIME_DIFF'] = (
            df_2005_2010['EVENT_' + var + '_TIME_DIFF'].fillna(
                method='ffill').fillna(0))
        df_2010_2015['EVENT_' + var + '_TIME_DIFF'] = (
            df_2010_2015['EVENT_' + var + '_TIME_DIFF'].fillna(
                method='ffill').fillna(0))

    # Sort dataframes
    df_2005_2010.sort_values(['ID', 'YEAR'], axis='rows', inplace=True)
    df_2010_2015.sort_values(['ID', 'YEAR'], axis='rows', inplace=True)
    # Save datasets for inspection
    df_2005_2010.to_pickle(ppj('OUT_DATA', 'panel_2005_2010_inspection.pkl'))
    df_2010_2015.to_pickle(ppj('OUT_DATA', 'panel_2010_2015_inspection.pkl'))
    # Get last row of each ID
    df_2005_2010 = df_2005_2010.groupby('ID', as_index=False).last()
    df_2010_2015 = df_2010_2015.groupby('ID', as_index=False).last()
    # Append two periods
    df_appended = df_2005_2010.append(df_2010_2015)

    return df_appended


@report_shape_of_dataframe
def drop_unused_columns_and_observations(df):
    # Drop columns
    unused_columns = []
    unused_columns += ['ID_ORIGINAL_HH']
    unused_columns += ['YEAR_2005_2010', 'YEAR_2005_2010_SUM',
                       'YEAR_2010_2015', 'YEAR_2010_2015_SUM']
    unused_columns += [i for i in df if 'SY' in i]
    unused_columns += [i for i in df if 'PY' in i]
    unused_columns += [i for i in df if '_MONTH' in i]
    unused_columns += [i for i in df if 'BEFORE_INTERVIEW' in i]
    unused_columns += ['REASON_JOB_TERMINATED',
                       'REASON_JOB_TERMINATED_SHIFTED']
    unused_columns += ['EVENT_' + i for i in EVENT_VARIABLES]
    unused_columns += ['ID_MOTHER', 'MOTHER_PREGNANT_AT_PQ_YEAR']
    unused_columns += ['EDUCATION_GROUPS_CASMIN', 'EDUCATION_GROUPS_ISCED11',
                       'YEARS_EDUCATION']
    df.drop(unused_columns, axis='columns', inplace=True)
    # Drop all 25 observations with NaNs in EDUCATION_GROUPS_ISCED97
    df.dropna(subset=['EDUCATION_GROUPS_ISCED97'], axis='rows', inplace=True)
    # Create dummies for events
    for i in EVENT_VARIABLES:
        df['EVENT_' + i] = (df['EVENT_' + i + '_COUNT'] != 0)
    # Restore dtypes after groupby.Groupby.last() operation
    df.GENDER = df.GENDER.astype('category')
    df.MARITAL_STATUS = df.MARITAL_STATUS.astype('category')
    df.EMPLOYMENT_STATUS = df.EMPLOYMENT_STATUS.astype('category')
    # Sort values
    df.sort_values(['ID', 'YEAR'], axis='rows', inplace=True)

    return df


if __name__ == '__main__':
    # Create a unaggregated panel from 2005 to 2015
    df = create_panel()
    # Clean common variables
    df = clean_common_variables(df)
    # Merge with ``edu_groups.pkl``
    df = merge_with_edu_groups(df)
    # Merge with ``edu_years``
    df = merge_with_edu_years(df)
    # Merge with ``migration.pkl``
    df = merge_with_migration(df)
    # Extract LOC for separate processing
    df = extract_loc(df)
    # Save dataframe for inspection
    df.to_pickle(ppj('OUT_DATA', 'panel_inspection_1.pkl'))
    # Merge with ``preg.pkl`` and ``dis.pkl``
    df = merge_with_preg_dis(df)
    # Clean event variables
    df = clean_event_variables(df)
    # Clean event data
    df = drop_unused_columns_and_observations(df)
    # Save the current dataframe
    df.to_pickle(ppj('OUT_DATA', 'panel.pkl'))
