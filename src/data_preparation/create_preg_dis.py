#!/usr/bin/env python

"""This module prepares ``bioagel.dta`` to extract events of unplanned
pregnancy and birth of children with disorders.

"""

import pandas as pd
from collections import OrderedDict
import numpy as np
from bld.project_paths import project_paths_join as ppj


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

VARIABLE_DICT_BIOAGEL = {
    # 'hhnr': 'ID_ORIGINAL_HH',  # original household id
    'hhnrakt': 'ID_HH',  # current household ID
    'persnr': 'ID_CHILD',
    'persnre': 'ID_MOTHER',  # never changing personal ID of mother
    'syear': 'YEAR',  # survey year
    # 'paround': 'INT_PARENTS_AROUND',
    'prebeg': 'PREGNANCY_START',  # begin of pregnancy
    'preend': 'PREGNANCY_END',  # end of pregnancy
    # 'fathinhh': 'FATHER_IN_HH',  # father lives in hh
    'pregplan': 'PREGNANCY_UNPLANNED',  # pregnancy planned or unplanned
    'birthm': 'CHILD_BIRTH_MONTH',  # month of birth of child
    'birthy': 'CHILD_BIRTH_YEAR',  # year of birth of child
    'disord': 'CHILD_DISORDER',  # does the child have disorders
    # 'disordu_fid': 'CHILD_DISORDER_CONFIRMED',  # first U exam confirming
                                                  # disorder
    # 'disord1': 'DISORDERS_PERCEPTION',
    # 'disord2': 'DISORDERS_MOTOR',
    # 'disord3': 'DISORDERS_NEUROLOGIC',
    # 'disord4': 'DISORDERS_LANGUAGE',
    # 'disord5': 'DISORDERS_REGULATION',
    # 'disord6': 'DISORDERS_CHRONIC_ILLNESS',
    # 'disord7': 'DISORDERS_PHSYSICAL',
    # 'disord8': 'DISORDERS_MENTAL',
    # 'disord9': 'DISORDERS_OTHER',
    'pregy': 'MOTHER_PREGNANT_AT_PQ_YEAR',  # mother was pregnant when taking
                                            # personal questionnaire
    # 'pregmo': 'MOTHER_PREGNANT_AT_PQ_MONTH_PREGNANCY',  # month of pregnancy
                                                          # when taking pq
    # 'specned8_fid': 'CHILD_LL_DISEASE',  # long-lasting disease
}

RETAINED_COLUMNS_BIOAGEL = list(VARIABLE_DICT_BIOAGEL.keys())


def prepare_bioagel():
    # Read the data and rename columns
    df = pd.read_stata(ppj('IN_DATA', 'bioagel.dta'),
                       columns=RETAINED_COLUMNS_BIOAGEL)
    df = df.rename(columns=VARIABLE_DICT_BIOAGEL)
    # Keep only data from 2004 to 2015, we need 2004 for the full pregnancy
    # spell
    df = df.loc[df.YEAR.between(2004, 2015)]
    df.sort_values(['ID_CHILD', 'YEAR'], axis='rows', inplace=True)
    df = df.groupby('ID_CHILD', as_index=False).first()

    # Extract year number to determine whether mother was pregnant at the time
    # of the interview
    df.MOTHER_PREGNANT_AT_PQ_YEAR = df.MOTHER_PREGNANT_AT_PQ_YEAR.str.extract(
        '\[(\d+)\]', expand=False)
    df.MOTHER_PREGNANT_AT_PQ_YEAR = df.MOTHER_PREGNANT_AT_PQ_YEAR.astype(float)

    # Convert values of CHILD_BIRTH_MONTH to real months
    child_birth_month_values = list(df.CHILD_BIRTH_MONTH.unique())
    child_birth_month_dict = {k: MONTH_DICT[v] for v in MONTH_DICT
                              for k in child_birth_month_values
                              if k in v.casefold()}
    child_birth_month_dict.update({'[-1] keine Angabe': np.nan,
                                   '[-2] trifft nicht zu': np.nan})
    df.CHILD_BIRTH_MONTH.replace(child_birth_month_dict, inplace=True)
    df.CHILD_BIRTH_MONTH = df.CHILD_BIRTH_MONTH.astype('category')
    df.CHILD_BIRTH_MONTH.cat.set_categories(MONTH_DICT.values(), ordered=True,
                                            inplace=True)

    # Create PREGNANCY_START_YEAR and _MONTH
    pregstart = df.PREGNANCY_START.str.split(' ', n=2, expand=True).drop(
        0, axis=1).fillna(value=np.nan)
    pregstart = pregstart.rename(columns={1: 'PREGNANCY_START_YEAR',
                                          2: 'PREGNANCY_START_MONTH'})
    # Rename monthly values by comparing short values to the keys of MONTH_DICT
    # and then the values of MONTH_DICT to map them to MONTH_DICT values
    month_values = pregstart.PREGNANCY_START_MONTH.unique()
    month_values = month_values[pd.notnull(month_values)]
    correct_month_mapping = {k: MONTH_DICT[v] for v in MONTH_DICT
                             for k in month_values
                             if k.casefold() in v.casefold()}
    correct_month_mapping.update({k: MONTH_DICT[v] for v in MONTH_DICT
                                  for k in month_values if k.casefold()
                                  in MONTH_DICT[v].casefold()})
    pregstart.PREGNANCY_START_MONTH.replace(correct_month_mapping,
                                            inplace=True)

    # Create PREGNANCY_END_YEAR and _MONTH
    pregend = df.PREGNANCY_END.str.split(' ', n=2, expand=True).drop(
        0, axis=1).fillna(value=np.nan)
    pregend = pregend.rename(columns={1: 'PREGNANCY_END_YEAR',
                                      2: 'PREGNANCY_END_MONTH'})
    # Rename monthly values by comparing short values to the keys of MONTH_DICT
    # and then the values of MONTH_DICT to map them to MONTH_DICT values
    month_values = pregend.PREGNANCY_END_MONTH.unique()
    month_values = month_values[pd.notnull(month_values)]
    correct_month_mapping = {k: MONTH_DICT[v] for v in MONTH_DICT
                             for k in month_values
                             if (k.casefold() in v.casefold())}
    correct_month_mapping.update({k: MONTH_DICT[v] for v in MONTH_DICT
                                  for k in month_values
                                  if k.casefold() in MONTH_DICT[v].casefold()})
    pregend.PREGNANCY_END_MONTH.replace(correct_month_mapping, inplace=True)

    df = pd.concat([df, pregstart, pregend], axis='columns').drop(
        ['PREGNANCY_START', 'PREGNANCY_END'], axis='columns')
    # Cast new variables to the correct types and set ordered
    df.PREGNANCY_START_MONTH = df.PREGNANCY_START_MONTH.astype('category')
    df.PREGNANCY_START_MONTH.cat.set_categories(
        MONTH_DICT.values(), ordered=True, inplace=True)
    df.PREGNANCY_START_YEAR = pd.to_numeric(
        df.PREGNANCY_START_YEAR, errors='coerce')
    df.PREGNANCY_END_MONTH = df.PREGNANCY_END_MONTH.astype('category')
    df.PREGNANCY_END_MONTH.cat.set_categories(
        MONTH_DICT.values(), ordered=True, inplace=True)
    df.PREGNANCY_END_YEAR = pd.to_numeric(
        df.PREGNANCY_END_YEAR, errors='coerce')

    return df


def create_preg(df):
    # Identify an event of an unplanned pregnancy with the begin of the
    # pregnancy
    pregnancy_unplanned = {
        '[2] eher geplant': np.nan, '[1] eher ungeplant': True,
        '[3] erfolgte mit med. unterstuetzung': np.nan
    }
    df.PREGNANCY_UNPLANNED.replace(pregnancy_unplanned, inplace=True)
    # Replace PREGNANCY_UNPLANNED with the month of the begin of the pregnancy
    df.loc[df.PREGNANCY_UNPLANNED.notnull(),
           'PREGNANCY_UNPLANNED_MONTH'] = df.PREGNANCY_START_MONTH
    # Cast the variables to the appropriate types
    df.PREGNANCY_UNPLANNED_MONTH = df.PREGNANCY_UNPLANNED_MONTH.astype(
        'category')
    df.PREGNANCY_UNPLANNED_MONTH.cat.set_categories(
        MONTH_DICT.values(), ordered=True, inplace=True)

    # Copy the necessary content
    preg = df[['ID_HH', 'ID_MOTHER', 'PREGNANCY_UNPLANNED_MONTH',
               'PREGNANCY_START_YEAR', 'MOTHER_PREGNANT_AT_PQ_YEAR']].copy()
    # Restrict to events
    preg = preg.loc[preg.PREGNANCY_UNPLANNED_MONTH.notnull()]
    # Rename columns to match panel
    preg_columns = {'PREGNANCY_START_YEAR': 'YEAR'}
    preg = preg.rename(columns=preg_columns)
    # Drop pregnancies which begin in 2004
    preg = preg.loc[preg.YEAR.between(2005, 2015)]
    # Drop pregnancies which were known at the time of the interview in 2005
    preg = preg.loc[~(preg.MOTHER_PREGNANT_AT_PQ_YEAR == 2005)]
    # We cannot drop pregnancies not known at the time of the pq in 2015 since
    # MOTHER_PREGNANT_AT_PQ_YEAR contains too much NaNs. The rest has to be
    # examined in comparison to INT_MONTH

    # There are 16 siblings with disabilities in our sample which leads to
    # the problem of two occurrences per ID and YEAR. Unfortunately, we cannot
    # handle this cases right now. Therefore, we drop the duplicates and save
    # them in another file to adjust the counts of events later.
    preg.loc[preg.duplicated(subset=['ID_MOTHER', 'YEAR'])].to_pickle(ppj(
        'OUT_DATA', 'preg_siblings.pkl'))
    preg.drop_duplicates(subset=['ID_MOTHER', 'YEAR'], inplace=True)

    # Save data
    preg.to_pickle(ppj('OUT_DATA', 'preg.pkl'))


def create_dis(df):
    # Identify an event of an unplanned pregnancy with month of birth
    child_disorder = {
        '[1] ja, bei U(1-6) Untersuchung': True,
        '[2] ja, bei anderer Untersuchung': True,
        '[3] nein': np.nan,
    }
    df.CHILD_DISORDER.replace(child_disorder, inplace=True)
    # Replace CHILD_DISORDER with the month of birth
    df.loc[df.CHILD_DISORDER.notnull(),
           'CHILD_DISORDER_MONTH'] = df.CHILD_BIRTH_MONTH
    # Cast variable to the appropriate type
    df.CHILD_DISORDER_MONTH = df.CHILD_DISORDER_MONTH.astype('category')
    df.CHILD_DISORDER_MONTH.cat.set_categories(
        MONTH_DICT.values(), ordered=True, inplace=True)

    # Copy the necessary content
    dis = df[['ID_HH', 'ID_MOTHER', 'CHILD_DISORDER_MONTH',
              'PREGNANCY_END_YEAR', 'MOTHER_PREGNANT_AT_PQ_YEAR']].copy()
    # Restrict to events
    dis = dis.loc[dis.CHILD_DISORDER_MONTH.notnull()]
    # Rename columns to match panel
    dis_columns = {'PREGNANCY_END_YEAR': 'YEAR'}
    dis = dis.rename(columns=dis_columns)
    # Drop births in 2004
    dis = dis.loc[dis.YEAR.between(2005, 2015)]
    # There are 9 siblings with disabilities in our sample which leads to
    # the problem of two occurrences per ID and YEAR. Unfortunately, we cannot
    # handle this cases right now. Therefore, we drop the duplicates and save
    # them in another file to adjust the counts of events later.
    dis.loc[dis.duplicated(subset=['ID_MOTHER', 'YEAR'])].to_pickle(ppj(
        'OUT_DATA', 'dis_siblings.pkl'))
    dis.drop_duplicates(subset=['ID_MOTHER', 'YEAR'], inplace=True)

    dis.to_pickle(ppj('OUT_DATA', 'dis.pkl'))


if __name__ == '__main__':
    # Prepare ``bioagel.dta``
    df = prepare_bioagel()
    # Create ``preg.pkl``
    create_preg(df)
    # Create ``dis.pkl``
    create_dis(df)
