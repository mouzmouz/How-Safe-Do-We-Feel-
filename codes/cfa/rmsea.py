import numpy as np
import pandas as pd
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser
import semopy 


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


# define the model
model_dict = {
    'Identity': ['identity_items_6_2', 'identity_items_8_2', 'identity_items_10_2', 'identity_items_11_2', 'identity_items_13'],
    'Traveling_Alone': ['social_int._10', 'social_int._13', 'q115_2'],
    'Perceived_High_Risk_Content': ['immedi._env._6', 'immedi._env._7', 'immedi._env._8'],
    'Interactions_with_Strangers': ['social_int._4', 'social_int._5']
}

# construct the model description string
model_spec = ""

for latent, observeds in model_dict.items():
    model_spec += f"{latent} =~ " + " + ".join(observeds) + "\n"

# create a model instance and load the dataset
model = semopy.Model(model_spec)
model.load_dataset(df)

# fit the model to your data
model.fit()
# inspect the model to get fit indices
res = semopy.calc_stats(model)
# all results
print(res)
