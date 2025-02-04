import numpy as np
import pandas as pd
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser
import pingouin as pg

file_path = 'Final_Survey.xlsx' 
df = pd.read_excel(file_path, sheet_name='Final_Survey')

# convert column names to lower-case for easier handling
df.columns = df.columns.str.lower()

# delete non-needed columns
df = df.loc[:, ~df.columns.isin(['startdate', 'enddate', 'status', 'ipaddress', 'recipientlastname', 'recipientfirstname',
       'recipientemail', 'externalreference', 'locationlatitude',
       'locationlongitude','progress', 'duration (in seconds)', 'finished', 'recordeddate',
       'responseid', 'distributionchannel', 'userlanguage', 'information sheet', 'age', 'gender', 'gender_4_text' ])]

# get rid of likert scale strings and replace with integers 1-2-3-4-5-6-7
replacer = {'Strongly disagree': 1, 'Disagree': 2,'Somewhat disagree': 3,
            'Neither agree nor disagree': 4,
            'Somewhat agree': 5, 'Agree':6,'Strongly agree': 7}
df= df.replace(replacer)


# Cronbach’s Alpha
factor1_items = df[['identity_items_6_2', 'identity_items_8_2', 'identity_items_10_2', 
                    'identity_items_11_2', 'identity_items_13']]
alpha_factor1 = pg.cronbach_alpha(data=factor1_items)
print("\na_factor1 = ", alpha_factor1)

factor2_items = df[['social_int._10', 'social_int._13', 'q115_2']]
alpha_factor2 = pg.cronbach_alpha(data=factor2_items)
print("\na_factor2 = ", alpha_factor2)

factor3_items = df[['immedi._env._6', 'immedi._env._7', 'immedi._env._8']]
alpha_factor3 = pg.cronbach_alpha(data=factor3_items)
print("\na_factor3 = ", alpha_factor3)

factor4_items = df[['social_int._4', 'social_int._5']]
alpha_factor4 = pg.cronbach_alpha(data=factor4_items)
print("\na_factor4 = ", alpha_factor4)

scale_items = df[['identity_items_6_2', 'identity_items_8_2','identity_items_10_2', 
                    'identity_items_11_2', 'identity_items_13', 'social_int._10', 'social_int._13', 'q115_2',
                    'immedi._env._6', 'immedi._env._7', 'immedi._env._8', 'social_int._4', 'social_int._5'
                    ]]
alpha_scale = pg.cronbach_alpha(data=scale_items)
print("\na_scale = ", alpha_scale)

# model specification
model_dict = {
    'Identity': ['identity items_6_2', 'identity items_8_2', 'identity items_10_2', 'identity items_11_2', 'identity items_13'],
    'Traveling Alone': ['social int._10', 'social int._13', 'q115_2'],
    'Perceived High-Risk Content': ['immedi. env._6', 'immedi. env._7', 'immedi. env._8'],
    'Interactions with Strangers': ['social int._4', 'social int._5']
}

# parse the model specification
model_spec = ModelSpecificationParser.parse_model_specification_from_dict(df, model_dict)

# fit the CFA model
cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
cfa.fit(df.values)

# unstandardized loadings
loadings = cfa.loadings_

# standard deviations of observed variables
observed_std = df.std().values

# standard deviations of latent factors (factor variance is typically 1 in standardized CFA)
latent_std = np.ones(loadings.shape[1])

# standardize loadings
standardized_loadings = loadings * latent_std / observed_std[:, None]

print("\nUnstandardized Loadings:")
print(loadings)

print("\nStandardized Loadings:")
print(standardized_loadings)
