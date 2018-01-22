# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:37:35 2018

@author: mywork
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy.contrasts import Treatment
from statsmodels.formula.api import ols
# from bld.project_paths import project_paths_join as ppj

# Import pickle
df = pd.read_pickle('panel.pkl')
print(df)
loc_data = pd.read_pickle('loc_pca.pkl')

# Create dummies.
df['GENDER'] = df['GENDER'].replace(['Female', 'Male'],[0, 1]) 
df['EMPLOYMENT_STATUS'] = df['EMPLOYMENT_STATUS'].replace(['Not Employed', 'Employed', 'Other'],[1, 2, 3]) 
df['MARITAL_STATUS'] = df['MARITAL_STATUS'].replace(['Single', 'Relationship'],[0, 1])

list_events = ['EVENT_DEATH_CHILD', 'EVENT_DEATH_FATHER', 'EVENT_DEATH_HH_PERSON',
               'EVENT_DEATH_MOTHER', 'EVENT_DEATH_PARTNER', 'EVENT_DIVORCED',
               'EVENT_HH_COMP_CHANGE', 'EVENT_LAST_JOB_ENDED', 'EVENT_SEPARATED']

booleanDictionary = {True: 1, False: 0}

for i in list_events:
    df['{}'.format(i)] = df['{}'.format(i)].replace(booleanDictionary)
    
# Separate loc data for the three different years. 
loc_years = [2005, 2010, 2015]
loc_data_2005, loc_data_2010, loc_data_2015 = [
        loc_data.loc[loc_data['YEAR'] == i] for i in loc_years]

# Calculate differences of the first and second axis values between 2005/2010 
# and 2010/2015.
loc_data_2010 = loc_data_2010.set_index('ID')
loc_data_2005 = loc_data_2005.set_index('ID')
loc_data_2015 = loc_data_2015.set_index('ID')
diff_05_10 = loc_data_2010[loc_data_2010.columns[0:2]].subtract(loc_data_2005[loc_data_2005.columns[0:2]], axis=0)
diff_10_15 = loc_data_2015[loc_data_2015.columns[0:2]].subtract(loc_data_2010[loc_data_2010.columns[0:2]], axis=0)

# Separate panel data for the three different years. 
df_years = [2010, 2015]
df_2010, df_2015 = [
        df.loc[df['YEAR'] == i] for i in df_years]

diff_05_10 = diff_05_10.dropna()
diff_10_15 = diff_10_15.dropna()

# Join the panel data with the difference in the two axes. 
df_2010 = df_2010.set_index('ID')
df_2010 = df_2010.join(diff_05_10)

df_2015 = df_2015.set_index('ID')
df_2015 = df_2015.join(diff_10_15)

df = pd.concat([df_2010, df_2015])
df = df.reset_index()
df = df.sort_values(by=['ID','YEAR'])

# Form 8 age groups. Each group encompasses about one decade.
df['AGE'] = pd.cut(df['AGE'], 8, labels=[1, 2, 3, 4, 5, 6, 7, 8])


# Save the dataframe which is needed for the regressions as a pickle.
df.to_pickle('panel_reg.pkl')




