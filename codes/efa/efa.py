import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import pingouin as pg

#read the qualtrics file
df = pd.read_csv('qualtrics_all.csv')


### CLEANING THE FILE ###

# convert column names to lower-case for easier handling
df.columns = df.columns.str.lower()


####################################################
# func to rename dublicate column names
def rename_duplicates(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index.tolist()
        for i in range(len(dup_indices)):
            cols[dup_indices[i]] = f"{dup}_{i+1}"
    df.columns = cols
    return df

# Rename duplicate columns
df = rename_duplicates(df)
#####################################################

# deleting the people that failed the attention checks
deleted_ids_video_check = df[df['video check'] != 'Yes']['id'].tolist()
#print("deleted because of video check: ", deleted_ids_video_check)
df = df.drop(df[df['video check'] != 'Yes'].index)
deleted_ids_prev_exp = df[df['prev. exp._5'] != 'Strongly disagree']['id'].tolist()
#print("\ndeleted because of pacific ocean check: ", deleted_ids_prev_exp)
df = df.drop(df[df['prev. exp._5'] != 'Strongly disagree'].index)
deleted_ids_social_int = df[df['social int._7'] != 'Agree']['id'].tolist()
#print("\ndeleted because of 'agree' check: ", deleted_ids_social_int)
df = df.drop(df[df['social int._7'] != 'Agree'].index)

# Combine all deleted IDs in a single list
all_deleted_ids = deleted_ids_video_check + deleted_ids_prev_exp + deleted_ids_social_int

df = df.drop(columns=['identity items_12_1','identity items_1_2', 'identity items_2_2']) # not directly relevant to pps 


df = df.drop(columns=['identity items_1_1','identity items_2_1', 'identity items_3_1', 'identity items_6_1', 
                      'identity items_5_1','identity items_7_1','identity items_8_1', 'identity items_9_1']) # too general  

# delete non-needed columns
df = df.loc[:, ~df.columns.isin(['startdate', 'enddate', 'status', 'ipaddress', 'recipientlastname', 'recipientfirstname',
       'recipientemail', 'externalreference', 'locationlatitude',
       'locationlongitude','progress', 'duration (in seconds)', 'finished', 'recordeddate',
       'responseid', 'distributionchannel', 'userlanguage',
       'information sheet', 'id', 'video check' , 'prev. exp._5', 'social int._7' ])]

df = df.drop(columns=['prev. exp._4']) #same question exists

# get rid of likert scale strings and replace with integers 1-2-3-4-5-6-7
replacer = {'Strongly disagree': 1, 'Disagree': 2,'Somewhat disagree': 3,
            'Neither agree nor disagree': 4,
            'Somewhat agree': 5, 'Agree':6,'Strongly agree': 7}
df= df.replace(replacer)

# drop rows with empty items
df = df.dropna()
print("\nAfter cleaning the data, the remaining participants are participants: ",len(df))


# write final items in csv file
df.to_csv('df_clean.csv', sep=',', index=False, encoding='utf-8')


#### CHECK FOR STUITABILITY FOR EFA ####

print("\nChecking stuitability for EFA with Bartlett's test and KMO test...")

# Bartlett’s Test of Sphericity
chi_square_value, p_value = calculate_bartlett_sphericity(df)
print(f"\nBartlett’s test p-value: {p_value}")
if p_value < 0.05:
    print("Bartlett’s test is supportive of performing EFA")

# Kaiser-Meyer-Olkin (KMO) Test
kmo_all, kmo_model = calculate_kmo(df)
print(f"\nKMO Test: {kmo_model}")
# Interpret the KMO value
if kmo_model < 0.5:
    print("KMO value below 0.5 indicates that EFA may not be suitable")
elif 0.5 <= kmo_model < 0.6:
    print("KMO value between 0.5 and 0.6 suggests EFA may not be suitable, but it's borderline")
elif 0.6 <= kmo_model < 0.7:
    print("KMO value between 0.6 and 0.7 is mediocre, but EFA can be considered")
elif 0.7 <= kmo_model < 0.8:
    print("KMO value between 0.7 and 0.8 is good for EFA")
elif 0.8 <= kmo_model < 0.9:
    print("KMO value between 0.8 and 0.9 is great for EFA")
elif kmo_model >= 0.9:
    print("KMO value above 0.9 is *amazing* for EFA")


### DETERMINE THE NUMBER OF FACTORS ###

print("\n\nDetermining the number for factors with Eigenvalues...")

# initial factor analysis to get eigenvalues - no rotation
fa = FactorAnalyzer(rotation=None)
fa.fit(df)
ev, v = fa.get_eigenvalues()
print(f"Eigenvalues: {ev}")

n_factors = sum(ev > 1)
print("\nAccording to eigenvalues I need ", n_factors," factors")

# Scree Plot
plt.plot(range(1, len(ev) + 1), ev, marker='o')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.axhline(y=1, color='r', linestyle='--')
plt.savefig('scree_plot.png')
plt.show()


# Identify and remove items with communalities less than 0.2
communalities = fa.get_communalities()
communalities_df = pd.DataFrame(communalities, index=df.columns, columns=['Communalities'])
items_to_keep = communalities_df[communalities_df['Communalities'] >= 0.2].index
df_filtered = df[items_to_keep]

# factor analysis with rotation
fa_filtered = FactorAnalyzer(n_factors=n_factors, rotation='oblimin')
fa_filtered.fit(df_filtered)

# Get the factor loadings and new communalities
revised_loadings = fa_filtered.loadings_
revised_loadings_df = pd.DataFrame(revised_loadings, index=df.columns, columns=[f'Factor{i+1}' for i in range(revised_loadings.shape[1])])


### EFA ###
def EFA(n_factors, df, rotation='oblimin'):
    efa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    efa.fit(df)

    # Get factor loadings
    loadings = efa.loadings_

    # Interpret the results
    loadings_df = pd.DataFrame(loadings, index=df.columns, columns=[f'Factor{i+1}' for i in range(loadings.shape[1])])

    # Save the loadings to a CSV file
    loadings_df.to_csv('factor_loadings.csv', sep=';', encoding='utf-8')

    # Visualize the loadings
    plt.matshow(loadings, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Factors')
    plt.ylabel('Variables')
    plt.title('Factor Loadings Heatmap')
    #plt.show()

    # Get eigenvalues and create scree plot
    ev, v = efa.get_eigenvalues()
    plt.plot(range(1, len(ev) + 1), ev, marker='o')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.axhline(y=1, color='r', linestyle='--')
    #plt.show()

    return efa


# remove items that are considered redundant and (re)conduct EFA

df = df.drop(columns=['identity items_4_1', 'immedi. env._3', 'immedi. env._9', 'prev. exp._1',
                      'prev. exp._3', 'prev. exp._7', 'social int._9', 'q115_5', 'q115_6', 'q115_7',
                      'q115_11', 'q115_12', 'q115_4','q115_8', 'q115_9', 'q115_10', 'immedi. env._4']) 
                      # delete cross loadings and items without significant loadings (6)

df = df.drop(columns=['identity items_10_1', 'identity items_11_1']) 
                    # delete cross loadings and items without significant loadings<0.4 (5)


df = df.drop(columns=['social int._12','social int._11', 'social int._1', 'identity items_3_2', 'prev. exp._6',
                      'immedi. env._10', 'identity items_4_2', 'identity items_9_2', 'prev. exp._2', 'social int._3',
                      'identity items_5_2', 'immedi. env._1', 'immedi. env._2', 'immedi. env._5',
                      'social int._2', 'social int._6', 'social int._8',
                      'q115_3',
                      'identity items_7_2', 'identity items_12_2',
                      'q115_1']) # delete cross loadings and items without significant loadings<0.44 (4)


# conduct EFA 
factors = 4 # initially 6, then 5, now 4
efa = EFA(factors, df, 'oblimin')
print("\nThe final factors and their items are now written in factor_loadings.csv")
