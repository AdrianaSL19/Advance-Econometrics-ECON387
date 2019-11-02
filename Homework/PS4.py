# Adriana Sham
# 10/02/2019
#PS4

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

m_06_12 = pd.read_csv('/Users/adriana/Downloads/Math_2006_2012_All.csv')
m_06_12.columns = ['DBN', 'Grade', 'Year', 'Demographic', 'Number Tested', 'Mean Scale Score', 'Num Level 1',
                   'Pct Level 1', 'Num Level 2', 'Pct Level 2', 'Num Level 3', 'Pct Level 3', 'Num Level 4',
                   'Pct Level 4', 'Num Level 3 and 4', 'Pct Level 3 and 4']  # assign names to columns

m_13_17 = pd.read_csv('/Users/adriana/Downloads/Math_2013_2017_All.csv')
m_13_17.columns = ['DBN', 'Grade', 'Year', 'Demographic', 'Number Tested', 'Mean Scale Score', 'Num Level 1',
                   'Pct Level 1', 'Num Level 2', 'Pct Level 2', 'Num Level 3', 'Pct Level 3', 'Num Level 4',
                   'Pct Level 4', 'Num Level 3 and 4', 'Pct Level 3 and 4']  # assign names to columns

e_06_12 = pd.read_csv('/Users/adriana/Downloads/English_2006_2012_All.csv')
e_06_12.columns = ['DBN', 'Grade', 'Year', 'Demographic', 'Number Tested', 'Mean Scale Score', 'Num Level 1',
                   'Pct Level 1', 'Num Level 2', 'Pct Level 2', 'Num Level 3', 'Pct Level 3', 'Num Level 4',
                   'Pct Level 4', 'Num Level 3 and 4', 'Pct Level 3 and 4']  # assign names to columns

e_13_17 = pd.read_csv('/Users/adriana/Downloads/English_2013_2017_All.csv')
e_13_17.columns = ['DBN', 'Grade', 'Year', 'Demographic', 'Number Tested', 'Mean Scale Score', 'Num Level 1',
                   'Pct Level 1', 'Num Level 2', 'Pct Level 2', 'Num Level 3', 'Pct Level 3', 'Num Level 4',
                   'Pct Level 4', 'Num Level 3 and 4', 'Pct Level 3 and 4']  # assign names to columns
e_13_17[['Pct Level 1', 'Pct Level 2', 'Pct Level 3', 'Pct Level 4', 'Pct Level 3 and 4']] = \
    e_13_17[['Pct Level 1', 'Pct Level 2', 'Pct Level 3', 'Pct Level 4',
             'Pct Level 3 and 4']].mul(100, axis=0)

English_2006_2017_All = e_06_12.append(e_13_17, sort=False)
cols_eng = English_2006_2017_All.columns.drop('DBN', 'Demographic')
English_2006_2017_All[cols_eng] = English_2006_2017_All[cols_eng].apply(pd.to_numeric, errors='coerce',
                                                                        downcast='integer')

English_2006_2017_All[['Pct Level 1', 'Pct Level 2', 'Pct Level 3', 'Pct Level 4', 'Pct Level 3 and 4']] = \
    English_2006_2017_All[['Pct Level 1', 'Pct Level 2', 'Pct Level 3', 'Pct Level 4',
                           'Pct Level 3 and 4']].div(100, axis=0)
English_2006_2017_All.columns = [col + '_eng' if col != 'DBN' and col != 'Grade' and col != 'Year'
                                 else col for col in English_2006_2017_All.columns]

Math_2006_2017_All = m_06_12.append(m_13_17, sort=False)
cols_math = Math_2006_2017_All.columns.drop('DBN', 'Demographic')
Math_2006_2017_All[cols_math] = Math_2006_2017_All[cols_math].apply(pd.to_numeric, errors='coerce', downcast='integer')
Math_2006_2017_All[['Pct Level 1', 'Pct Level 2', 'Pct Level 3', 'Pct Level 4', 'Pct Level 3 and 4']] = \
    Math_2006_2017_All[['Pct Level 1', 'Pct Level 2', 'Pct Level 3', 'Pct Level 4',
                        'Pct Level 3 and 4']].div(100, axis=0)
Math_2006_2017_All.columns = [col + '_math' if col != 'DBN' and col != 'Grade' and col != 'Year'
                              else col for col in Math_2006_2017_All.columns]

ME_2006_2017_All = English_2006_2017_All.merge(Math_2006_2017_All, on=('DBN', 'Grade', 'Year'))
ME_2006_2017_All['borough'] = ME_2006_2017_All['DBN'].str[2:3].astype(str)
ME_2006_2017_All.describe()

ME_2006_2017_All.groupby(
    ['borough', 'Year', 'Grade']
).agg(
    {
        'Pct Level 3 and 4_math': [np.mean, np.std, min, max],
        'Pct Level 3 and 4_eng' : [np.mean, np.std, min, max]
    }
)


ME_2006_2017_All['district'] = ME_2006_2017_All['DBN'].str[:2].astype(int)

borough_new = pd.get_dummies(ME_2006_2017_All['borough'])
Year_new = pd.get_dummies(ME_2006_2017_All['Year'])
district_new = pd.get_dummies(ME_2006_2017_All['district'])
ME_2006_2017_All['const'] = 1

Y = ME_2006_2017_All['Pct Level 3 and 4_math']
math_scores_model = smf.ols(formula = 'Y ~ const + Year_new + borough_new + district_new', data=ME_2006_2017_All)
# result_math_scores_model = math_scores_model.fit()
# print(result_math_scores_model.summary())


Y_english = ME_2006_2017_All['Pct Level 3 and 4_eng']
english_scores_model = smf.ols(formula = 'Y_english ~ const + Year_new + borough_new + district_new',
                               data=ME_2006_2017_All)
# result_english_scores_model  = english_scores_model.fit()
# print(result_english_scores_model.summary())

# export_csv = ME_2006_2017_All.to_csv(r'/Users/adriana/Downloads/export_dataframe.csv', index=None, header=True)
