# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:00:44 2018

@author: mywork
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# from bld.project_paths import project_paths_join as ppj

# Import pickle
df = pd.read_pickle('panel_reg.pkl')

# Prepare categorials for inserting them in the regression.
levels_employment = [1,2,3]
contrast = Treatment(reference=0).code_without_intercept(levels_employment)
print(contrast.matrix)
print(contrast.matrix[df.EMPLOYMENT_STATUS-1,:][:20])

levels_age = [1,2,3,4,5,6,7,8]
contrast = Treatment(reference=0).code_without_intercept(levels_age)
print(contrast.matrix)
print(contrast.matrix[df.AGE-1,:][:20])

# Set up column with counts of all traumatic events. 
event_counts = df[counts].sum(axis=1)
event_counts_df = pd.DataFrame(event_counts, columns=['EVENT_COUNTS_ALL'])
df = pd.concat([event_counts_df, df], axis=1)

# Run first specification. 
first = ols("FIRST_AXIS ~ C(EMPLOYMENT_STATUS, Treatment) + C(AGE, Treatment) + ID + GENDER + MARITAL_STATUS + EVENT_DEATH_CHILD + EVENT_DEATH_FATHER + EVENT_DEATH_HH_PERSON + EVENT_DEATH_MOTHER + EVENT_DEATH_PARTNER + EVENT_DIVORCED + EVENT_HH_COMP_CHANGE + EVENT_LAST_JOB_ENDED + EVENT_SEPARATED", data=df)
res_first = first.fit()
print(res_first.summary())

counts = ['EVENT_DEATH_CHILD_COUNT', 'EVENT_DEATH_FATHER_COUNT',
          'EVENT_DEATH_HH_PERSON_COUNT', 'EVENT_DEATH_MOTHER_COUNT', 
          'EVENT_DEATH_PARTNER_COUNT', 'EVENT_DIVORCED_COUNT', 
          'EVENT_HH_COMP_CHANGE_COUNT', 'EVENT_LAST_JOB_ENDED_COUNT', 
          'EVENT_SEPARATED_COUNT']

# Run third specification.
third = ols("FIRST_AXIS ~ C(EMPLOYMENT_STATUS, Treatment) + C(AGE, Treatment) + ID + GENDER + MARITAL_STATUS + EVENT_COUNTS_ALL + I(EVENT_COUNTS_ALL**2)", data=df)
res_third = third.fit()
print(res_third.summary())

