#!/usr/bin/env python

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from bld.project_paths import project_paths_join as ppj


TABLE_CONVERSION_1 = {
                    'FIRST_FACTOR_DELTA': 'Change in Locus of Control',
                    'C(EVENT\_LAST\_JOB\_ENDED)[True]:EVENT\_LAST\_JOB\_ENDED\_TIME\_DIFF:C(EMPLOYMENT\_STATUS)[T.Not Employed]':'Displacement * Time Since Occurrence * Not Employed',
                    'C(EVENT\_LAST\_JOB\_ENDED)[True]:EVENT\_LAST\_JOB\_ENDED\_TIME\_DIFF:C(EMPLOYMENT\_STATUS)[T.Other]':'Displacement * Time Since Occurrence * Other Empl. Status',
                    'C(EVENT\_LAST\_JOB\_ENDED\_LIMITED)[True]:EVENT\_LAST\_JOB\_ENDED\_LIMITED\_TIME\_DIFF:C(EMPLOYMENT\_STATUS)[T.Not Employed]':'Exog. Displacement * Time Since Occurrence * Not Employed',
                    'C(EVENT\_LAST\_JOB\_ENDED\_LIMITED)[True]:EVENT\_LAST\_JOB\_ENDED\_LIMITED\_TIME\_DIFF:C(EMPLOYMENT\_STATUS)[T.Other]':'Exog. Displacement * Time Since Occurrence * Other Empl. Status',
                    'C(EVENT\_SEPARATED)[True]:EVENT\_SEPARATED\_TIME\_DIFF:C(MARITAL\_STATUS)[T.Single]':'Separation * Time Since Occurrence * Single',
                    'C(MARITAL\_STATUS)[T.Single]:C(EVENT\_DIVORCED)[True]:EVENT\_DIVORCED\_TIME\_DIFF':'Divorce * Single * Time Since Occurrence',
                    'C(MARITAL\_STATUS)[T.Single]:C(EVENT\_DIVORCED)[T.True]': 'Divorce * Single', 
                    'C(EVENT\_LAST\_JOB\_ENDED)[T.True]:C(EMPLOYMENT\_STATUS)[T.Not Employed]':'Displacement * Not Employed',
                    'C(EVENT\_LAST\_JOB\_ENDED)[T.True]:C(EMPLOYMENT\_STATUS)[T.Other]':'Displacement * Other Empl. Status',
                    'C(EVENT\_LAST\_JOB\_ENDED\_LIMITED)[T.True]:C(EMPLOYMENT\_STATUS)[T.Not Employed]':'Exog. Displacement * Not Employed',
                    'C(EVENT\_LAST\_JOB\_ENDED\_LIMITED)[T.True]:C(EMPLOYMENT\_STATUS)[T.Other]':'Exog. Displacement * Other Empl. Status',
                    'C(EVENT\_SEPARATED)[T.True]:C(MARITAL\_STATUS)[T.Single]':'Separation * Single',
                    'C(EVENT\_CHILD\_DISORDER)[True]:EVENT\_CHILD\_DISORDER\_TIME\_DIFF':'Child Disorders * Time Since Occurrence', 
                    'C(EVENT\_DEATH\_CHILD)[True]:EVENT\_DEATH\_CHILD\_TIME\_DIFF':'Child Death * Time Since Occurrence',
                    'C(EVENT\_DEATH\_FATHER)[True]:EVENT\_DEATH\_FATHER\_TIME\_DIFF':'Father Death * Time Since Occurrence',
                    'C(EVENT\_DEATH\_HH\_PERSON)[True]:EVENT\_DEATH\_HH\_PERSON\_TIME\_DIFF':'Household Person Death * Time Since Occurrence',
                    'C(EVENT\_DEATH\_MOTHER)[True]:EVENT\_DEATH\_MOTHER\_TIME\_DIFF':'Mother Death * Time Since Occurrence',
                    'C(EVENT\_DEATH\_PARTNER)[True]:EVENT\_DEATH\_PARTNER\_TIME\_DIFF':'Partner Death * Time Since Occurrence',
                    'C(EVENT\_DIVORCED)[True]:EVENT\_DIVORCED\_TIME\_DIFF':'Divorce * Time Since Occurrence',
                    'C(EVENT\_HH\_COMP\_CHANGE)[True]:EVENT\_HH\_COMP\_CHANGE\_TIME\_DIFF':'HH Composition Changed * Time Since Occurrence',
                    'C(EVENT\_LAST\_JOB\_ENDED)[True]:EVENT\_LAST\_JOB\_ENDED\_TIME\_DIFF':'Displacement * Time Since Occurrence',
                    'C(EVENT\_LAST\_JOB\_ENDED\_LIMITED)[True]:EVENT\_LAST\_JOB\_ENDED\_LIMITED\_TIME\_DIFF':'Exog. Displacement * Time Since Occurrence',
                    'C(EVENT\_LEGALLY\_HANDICAPPED\_PERC)[True]:EVENT\_LEGALLY\_HANDICAPPED\_PERC\_TIME\_DIFF':'Handicapped * Time Since Occurrence',
                    'C(EVENT\_PREGNANCY\_UNPLANNED)[True]:EVENT\_PREGNANCY\_UNPLANNED\_TIME\_DIFF':'Unplanned Pregnancy * Time Since Occurrence',
                    'C(EVENT\_SEPARATED)[True]:EVENT\_SEPARATED\_TIME\_DIFF':'Separation * Time Since Occurrence',
                    'C(EVENT\_CHILD\_DISORDER)[T.True]': 'Child Has Disorders',
                    'C(EVENT\_DEATH\_CHILD)[T.True]':'Death of Child',
                    'C(EVENT\_DEATH\_FATHER)[T.True]':'Death of Father',
                    'C(EVENT\_DEATH\_HH\_PERSON)[T.True]':'Death of HH Person',
                    'C(EVENT\_DEATH\_MOTHER)[T.True]':'Death of Mother',
                    'C(EVENT\_DEATH\_PARTNER)[T.True]':'Death of Partner',
                    'C(EVENT\_DIVORCED)[T.True]':'Divorce',
                    'C(EVENT\_HH\_COMP\_CHANGE)[T.True]':'HH Composition Changed',
                    'C(EVENT\_LAST\_JOB\_ENDED)[T.True]':'Displacement',
                    'C(EVENT\_LEGALLY\_HANDICAPPED\_PERC)[T.True]':'Legally Handicapped',
                    'C(EVENT\_PREGNANCY\_UNPLANNED)[T.True]':'Unplanned Pregnancy',
                    'C(EVENT\_SEPARATED)[T.True]':'Separation',
                    'C(EVENT\_LAST\_JOB\_ENDED\_LIMITED)[T.True]':'Exogenous Displacement'
                    }

TABLE_CONVERSION_2 = {
    'FIRST_FACTOR_DELTA': 'Change in Locus of Control',
    'I(EVENT\_COUNTS\_ALL ** 2)': 'Number of Traumata Squared',
    'EVENT\_COUNTS\_ALL': 'Number of Traumata'}


def main(df):

    list_counts = [
        i for i in df if 'EVENT' in i and 'COUNT' in i and 'PREVIOUS' not in i and 'LIMITED' not in i]
    # Set up column with counts of all traumatic events.
    df['EVENT_COUNTS_ALL'] = df[list_counts].sum(axis=1)
    # Replace all 0 values in net hh income with 2 as ln(0) is not possible in regression.
    df['HH_NET_INCOME_YEAR'] = df['HH_NET_INCOME_YEAR'].replace([0.0], [2.0])

    # Run first specification.
    model_1 = smf.ols('FIRST_FACTOR_DELTA ~ C(EDUCATION_GROUPS_ISCED97) + C(MIGRATION_STATUS) + C(AGE_GROUPS) + C(GENDER) + np.log(HH_NET_INCOME_YEAR) + C(EVENT_CHILD_DISORDER) + C(EVENT_CHILD_DISORDER):EVENT_CHILD_DISORDER_TIME_DIFF + C(EVENT_DEATH_CHILD) + C(EVENT_DEATH_CHILD):EVENT_DEATH_CHILD_TIME_DIFF +  C(EVENT_DEATH_FATHER) + C(EVENT_DEATH_FATHER):EVENT_DEATH_FATHER_TIME_DIFF + C(EVENT_DEATH_HH_PERSON) + C(EVENT_DEATH_HH_PERSON):EVENT_DEATH_HH_PERSON_TIME_DIFF+ C(EVENT_DEATH_MOTHER) + C(EVENT_DEATH_MOTHER):EVENT_DEATH_MOTHER_TIME_DIFF + C(EVENT_DEATH_PARTNER) + C(EVENT_DEATH_PARTNER):EVENT_DEATH_PARTNER_TIME_DIFF + C(MARITAL_STATUS):C(EVENT_DIVORCED) + C(EVENT_DIVORCED) + C(EVENT_DIVORCED):EVENT_DIVORCED_TIME_DIFF + C(MARITAL_STATUS):C(EVENT_DIVORCED):EVENT_DIVORCED_TIME_DIFF + C(MARITAL_STATUS) + C(EVENT_HH_COMP_CHANGE) + C(EVENT_HH_COMP_CHANGE):EVENT_HH_COMP_CHANGE_TIME_DIFF + C(EVENT_LAST_JOB_ENDED):C(EMPLOYMENT_STATUS) + C(EVENT_LAST_JOB_ENDED) + C(EVENT_LAST_JOB_ENDED):EVENT_LAST_JOB_ENDED_TIME_DIFF + C(EVENT_LAST_JOB_ENDED):EVENT_LAST_JOB_ENDED_TIME_DIFF:C(EMPLOYMENT_STATUS) + C(EMPLOYMENT_STATUS) + C(EVENT_LEGALLY_HANDICAPPED_PERC) + C(EVENT_LEGALLY_HANDICAPPED_PERC):EVENT_LEGALLY_HANDICAPPED_PERC_TIME_DIFF + C(EVENT_PREGNANCY_UNPLANNED) + C(EVENT_PREGNANCY_UNPLANNED):EVENT_PREGNANCY_UNPLANNED_TIME_DIFF + C(EVENT_SEPARATED):C(MARITAL_STATUS) + C(EVENT_SEPARATED) + C(EVENT_SEPARATED):EVENT_SEPARATED_TIME_DIFF + C(EVENT_SEPARATED):EVENT_SEPARATED_TIME_DIFF:C(MARITAL_STATUS)', df)
    res_1 = model_1.fit(cov_type='cluster', cov_kwds={
                        'groups': df['ID']}, use_t=True)
    
    #Run second specification.
    model_2 = smf.ols('FIRST_FACTOR_DELTA ~ C(EDUCATION_GROUPS_ISCED97) + C(MIGRATION_STATUS) + C(AGE_GROUPS) + C(GENDER) + np.log(HH_NET_INCOME_YEAR) + C(MARITAL_STATUS) + C(EVENT_CHILD_DISORDER) + C(EVENT_CHILD_DISORDER):EVENT_CHILD_DISORDER_TIME_DIFF + C(EVENT_DEATH_CHILD) + C(EVENT_DEATH_CHILD):EVENT_DEATH_CHILD_TIME_DIFF +  C(EVENT_DEATH_FATHER) + C(EVENT_DEATH_FATHER):EVENT_DEATH_FATHER_TIME_DIFF + C(EVENT_DEATH_HH_PERSON) + C(EVENT_DEATH_HH_PERSON):EVENT_DEATH_HH_PERSON_TIME_DIFF + C(EVENT_DEATH_MOTHER) + C(EVENT_DEATH_MOTHER):EVENT_DEATH_MOTHER_TIME_DIFF + C(EVENT_DEATH_PARTNER) + C(EVENT_DEATH_PARTNER):EVENT_DEATH_PARTNER_TIME_DIFF + C(EVENT_HH_COMP_CHANGE) + C(EVENT_HH_COMP_CHANGE):EVENT_HH_COMP_CHANGE_TIME_DIFF + C(EVENT_LAST_JOB_ENDED_LIMITED):C(EMPLOYMENT_STATUS) + C(EVENT_LAST_JOB_ENDED_LIMITED) + C(EVENT_LAST_JOB_ENDED_LIMITED):EVENT_LAST_JOB_ENDED_LIMITED_TIME_DIFF + C(EVENT_LAST_JOB_ENDED_LIMITED):EVENT_LAST_JOB_ENDED_LIMITED_TIME_DIFF:C(EMPLOYMENT_STATUS) + C(EMPLOYMENT_STATUS) + C(EVENT_LEGALLY_HANDICAPPED_PERC) + C(EVENT_LEGALLY_HANDICAPPED_PERC):EVENT_LEGALLY_HANDICAPPED_PERC_TIME_DIFF + C(EVENT_PREGNANCY_UNPLANNED) + C(EVENT_PREGNANCY_UNPLANNED):EVENT_PREGNANCY_UNPLANNED_TIME_DIFF', df)
    res_2 = model_2.fit(cov_type='cluster', cov_kwds={
                        'groups': df['ID']}, use_t=True)
    
    # Run third specification.
    model_3 = smf.ols('FIRST_FACTOR_DELTA ~ C(EDUCATION_GROUPS_ISCED97) + C(EMPLOYMENT_STATUS) + C(MIGRATION_STATUS) + C(AGE_GROUPS) + C(GENDER) + np.log(HH_NET_INCOME_YEAR) + C(MARITAL_STATUS) + EVENT_COUNTS_ALL + I(EVENT_COUNTS_ALL**2)', df)
    res_3 = model_3.fit(cov_type='cluster', cov_kwds={
                        'groups': df['ID']}, use_t=True)

    ## Run second specification and replace time diff with dummies for yearly groups.
    #model_4 = smf.ols('FIRST_FACTOR_DELTA ~ C(EDUCATION_GROUPS_ISCED97) + C(MIGRATION_STATUS) + C(AGE_GROUPS) + C(GENDER) + np.log(HH_NET_INCOME_YEAR) + C(MARITAL_STATUS) + C(EVENT_CHILD_DISORDER) + C(EVENT_CHILD_DISORDER):C(EVENT_CHILD_DISORDER_TIME_DIFF_GROUP)+ C(EVENT_DEATH_CHILD) + C(EVENT_DEATH_CHILD):C(EVENT_DEATH_CHILD_TIME_DIFF_GROUP) +  C(EVENT_DEATH_FATHER) + C(EVENT_DEATH_FATHER):C(EVENT_DEATH_FATHER_TIME_DIFF_GROUP)+ C(EVENT_DEATH_HH_PERSON) + C(EVENT_DEATH_HH_PERSON):C(EVENT_DEATH_HH_PERSON_TIME_DIFF_GROUP) + C(EVENT_DEATH_MOTHER) + C(EVENT_DEATH_MOTHER):C(EVENT_DEATH_MOTHER_TIME_DIFF_GROUP)+ C(EVENT_DEATH_PARTNER) + C(EVENT_DEATH_PARTNER):C(EVENT_DEATH_PARTNER_TIME_DIFF_GROUP)+ C(EVENT_HH_COMP_CHANGE) + C(EVENT_HH_COMP_CHANGE):C(EVENT_HH_COMP_CHANGE_TIME_DIFF_GROUP) + C(EVENT_LAST_JOB_ENDED_LIMITED):C(EMPLOYMENT_STATUS) + C(EVENT_LAST_JOB_ENDED_LIMITED) + C(EVENT_LAST_JOB_ENDED_LIMITED):C(EVENT_LAST_JOB_ENDED_LIMITED_TIME_DIFF_GROUP) + C(EVENT_LAST_JOB_ENDED_LIMITED):C(EVENT_LAST_JOB_ENDED_LIMITED_TIME_DIFF_GROUP):C(EMPLOYMENT_STATUS) + C(EMPLOYMENT_STATUS)+ C(EVENT_LEGALLY_HANDICAPPED_PERC) + C(EVENT_LEGALLY_HANDICAPPED_PERC):C(EVENT_LEGALLY_HANDICAPPED_PERC_TIME_DIFF_GROUP) + C(EVENT_PREGNANCY_UNPLANNED) + C(EVENT_PREGNANCY_UNPLANNED):C(EVENT_PREGNANCY_UNPLANNED_TIME_DIFF_GROUP)', df)
    #res_4 = model_4.fit(cov_type='cluster', cov_kwds={
    #                    'groups': df['ID']}, use_t=True)


    # Write summaries from the first three regressions to latex tables, replace
    # original variable names and include only relevant rows.
    reg_table_1 = res_1.summary().as_latex()
    reg_table_1 = reg_table_1.split('\n')
    reg_table_1 = ' \n '.join(reg_table_1[0:17] + reg_table_1[31:38] + reg_table_1[39:41] + 
                              reg_table_1[43:50]  + reg_table_1[52:53] + reg_table_1[54:55] + 
                              reg_table_1[56:57] + reg_table_1[58:59] + reg_table_1[60:61] + 
                              reg_table_1[62:63] + reg_table_1[64:65] + reg_table_1[66:67] + 
                              reg_table_1[68:69] + reg_table_1[70:71] + reg_table_1[72:73] + 
                              reg_table_1[74:75] + reg_table_1[76:77] + reg_table_1[78:79] + 
                              reg_table_1[80:81] + reg_table_1[82:83]  + reg_table_1[91:])
    for key in TABLE_CONVERSION_1:
        reg_table_1 = reg_table_1.replace(key, TABLE_CONVERSION_1[key])
    table = open(ppj('OUT_TABLES', 'reg_table_1.tex'), 'w')
    table.writelines(reg_table_1)
    table.close()
    print(reg_table_1)
    
    
    reg_table_2 = res_2.summary().as_latex()
    reg_table_2 = reg_table_2.split('\n')
    reg_table_2 = ' \n '.join(reg_table_2[0:17] + reg_table_2[32:40] + reg_table_2[42:46] + 
                              reg_table_2[48:49] + reg_table_2[50:51] + reg_table_2[52:53] + 
                              reg_table_2[54:55] + reg_table_2[56:57] + reg_table_2[58:59] + 
                              reg_table_2[60:61] + reg_table_2[62:63] + reg_table_2[64:65] + 
                              reg_table_2[66:67] + reg_table_2[68:69] + reg_table_2[70:71] + 
                              reg_table_2[78:])
    for key in TABLE_CONVERSION_1:
        reg_table_2 = reg_table_2.replace(key, TABLE_CONVERSION_1[key])
    table = open(ppj('OUT_TABLES', 'reg_table_2.tex'), 'w')
    table.writelines(reg_table_2)
    table.close()

    reg_table_3 = res_3.summary().as_latex()
    reg_table_3 = reg_table_3.split('\n')
    reg_table_3 = ' \n '.join(
        reg_table_3[0:17] + reg_table_3[35:39] + reg_table_3[47:])
    for key in TABLE_CONVERSION_2:
        reg_table_3 = reg_table_3.replace(key, TABLE_CONVERSION_2[key])
    table = open(ppj('OUT_TABLES', 'reg_table_3.tex'), 'w')
    table.writelines(reg_table_3)
    table.close()


if __name__ == '__main__':
    # Import pickle
    df = pd.read_pickle(ppj('OUT_ANALYSIS', 'panel_reg.pkl'))
    main(df)
