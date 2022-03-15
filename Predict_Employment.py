# %% [markdown]
# ### Mohamad Quteifan
# 
# 
# # Problem Statement
# The journey to employment is not an easy one. We spend our time learning and developing skills to eventually get a job. The research conducted in the following notebook will focus on predicting if a fellow will be placed and the time it will take for a fellow to be placed. Throughout the notebook, I will derive insights around fellows and create two models to effectively predict if and when a fellow will be placed.
# 
# That being said, there are two models we are going to create in this case study: 
# 
# 1. Classification model: Whether or not the fellow will be placed.
# 2. Regression model: Predict the length that it will take a fellow to find placement. 
# 
# The other questions that I will tackle in the research:
# 1. Overall placement, how many individuals in the program were placed? 
# 2. Pathrise Placement, did pathrise have an impact?
# 3. What is the education of the fellows that were placed, did the education impact placement?
# 4. Duration until placement, how long does it take a fellow to be placed?
# 

# %% [markdown]
# ### Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
from scipy import stats
!pip install missingno
import warnings
import time
import pickle
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, f1_score, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from scipy import stats
!pip install pandas_profiling
import pandas_profiling

warnings.filterwarnings('ignore')
import pip
pip.main(['install', 'xgboost'])
import sys
!{sys.executable} -m pip install xgboost
import xgboost as xgb
import missingno as msno 
%matplotlib inline
import seaborn as sns
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score

from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

# %% [markdown]
# # Exploratory Data Analysis

# %% [markdown]
# ### Import data and review structure/data

# %%
df = pd.read_excel("Data_Pathrise.xlsx")

# %%
df

# %%
print("The shape of the data:\n",
      df.shape, 
      "\nThe first 5 rows are:\n", 
      df.head(), 
      "\nThe last 5 rows are:\n",
      df.tail(), 
      "\nThe column names are:\n",
      df.columns)
      


# %%
df.profile_report() 

# %% [markdown]
# ### Variable analysis

# %%
print("Pathrise status:\n", df["pathrise_status"].value_counts(),"primary track:\n",
      df["primary_track"].value_counts(),"cohort tag:\n", df["cohort_tag"].value_counts(),
      "program_duration_days:\n", df["program_duration_days"].value_counts().index,
      "placed:\n", df["placed"].value_counts(),
      "Employment Status:\n", df["employment_status "].value_counts(),
      "highest Level of education:\n", df["highest_level_of_education"].value_counts(),
      "Length of the job search:\n", df["length_of_job_search"].value_counts(),
      "Biggest challenge during the job search process:\n", df["biggest_challenge_in_search"].value_counts(),
      "Professional Experience:\n", df["professional_experience"].value_counts(),
      "Work authorization status (citizenship):\n", df["work_authorization_status"].value_counts(),
      "The number of interviews:\n", df["number_of_interviews"].value_counts(),
      "number_of_applications:\n", df["number_of_applications"].value_counts(),
      "gender:\n", df["gender"].value_counts(),
      "race:\n", df["race"].value_counts()
         
     )

# %% [markdown]
# ### Quick Analysis on the Review of the data

# %% [markdown]
# The data file imported contains 2544 rows and 16 columns. The columns of the data are: 
# 
# #1. id: the identification of the individual, each id represents an individual. It consists of numeric values starting with 1. 
# 
# #2. pathrise_status: The variable represent the pathrise status of the individual. It is a text variable (values consists of textual data) and there are 9 different values.
# 
# #3. primary_track: The variable represents the individuals career path/track. It is a text variable (values consists of textual data) and there are 6 different values.
# 
# #4. cohort_tag: The variable represents the class(cohort) the individual joined. It is a datetime variable, consists of text and numeric values. The variable will unlikily be used when modeling. The variable contains 47 different values. 
# 
# #5. program_duration_days: The length that the individual was part of the program. It consists of numeric values. There are a total of 411 different values. 
# 
# #6. placed: The variable represents if the individual was successfully placed in a position. It consists of numeric values and there are 2 different values. 0 = not placed 1 = placed. 
# 
# #7. employment_status: The variable represents the employement status of the individual prior to joining pathrise. It consists of text data and there are 5 different values. The variable contains a space in the column and that needs to be corrected.
# 
# #8. highest_level_of_education: The variable represents the education of the individual prior to joining pathrise. It consists of text data and there are 7 different values. This will be used in the model more than likily.
# 
# #9. length_of_job_search: The variable represents the length of the job search of the individual. It consists of text data and there are 5 different values. 
# 
# #10. biggest_challenge_in_search: The variable represents the biggest challenge the individual had prior to joining pathrise. It consists of text data and there are 10 different values. 
# 
# #11. professional_experience: The variable represents the professional experience of the individual prior to joining pathrise. It consists of text data and there are 4 different values.
# 
# #12. work_authorization_status: The variable represents the work authorization status(citizenship/authorization). It consists of text data and there are 9 different values. 
# 
# #13. number_of_interviews: The variable represents the number of interviews an individual had prior to joining pathrise. It consists of numeric values and there are 21 values.
# 
# #14. number_of_applications: The vairable represents the number of applciations that the individual filled prior to joining pathrise. It consists of numerical values and there are 41 different values. 
# 
# #15. gender: The variable represents the gender(sex) of the individual. It consist of text data and there are 4 different values.  
# 
# #16. race: The variables represents the race of the individual. It consists of text data and there are 9 different values. 

# %%
df

# %% [markdown]
# ## Placement Analysis

# %% [markdown]
# ### Part 1: Overall placement, how many individuals in the program were placed? 

# %%
placed = df["placed"]

placed= placed.replace([0],"not_placed")
placed = placed.replace([1],"placed")


#Check if there are any missing values within the variable
print("Are there any NaN or empty values in the column?", placed.isnull().values.any(), "\nThere are: ",
placed.isnull().values.sum(), "in the column")

placed

# %% [markdown]
# The good thing about the column "placed" is that there are no NaN or empty values. This is not the case for the majority of the data frame, but that will be looked at in the following steps. The purpose of this portion of the research is to calculate the percentage of individuals getting placed. I am converting the numeric values into text values for readability, easier to review visual when its placed vs. not_placed compared to 1 vs. 0.  

# %% [markdown]
# ### Visualize the distribution of placement

# %%
plt.hist(placed, bins=2, histtype='step')
# Label the axes
plt.xlabel('Placement')
plt.ylabel('Count')
plt.title('Pathrise Placement')
# Show the figure
plt.show()


# %%
plt.savefig('place_fig11.PDF') #save plot to desktop for research paper.

# %% [markdown]
# Statistics

# %%
print(placed.value_counts())
placements = 956
not_placed = 1588
total_individuals = 2544

print("Out of all the individuals included in the data,",((placements/total_individuals)*100),"% were placed.")


# %% [markdown]
# ### Quick placement analysis 
# Taking a glance at the distribution(histogram) we can see that more individuals were not placed. This information may be misleading since many individuals withdraw from the program or did not participate in the program. We can conclude that approximately 37.6% of individuals were placed based on this information. In the following steps, we will look deeper into the difference between Pathrise individuals who did not withdraw from the program. The focus will be to determine if Pathrise had an impact on placement. 

# %% [markdown]
# ### Part 2: Pathrise Placement, did pathrise have an impact? 

# %% [markdown]
# The important part is isolating the individuals who did not withdraw from the program and to analyze the success rate of Pathrise fellows.

# %%
df["pathrise_status"].value_counts()

# %% [markdown]
# #### Placed and active are considered part of the pathrise program, withdrawn, withdrawn (Trail), MIA, Deferred, and break are all individuals who did not participate in the program and are not considered fellows. Closed Lost are the individuals who went through the program and did not gain placement. 

# %%
path = df["pathrise_status"]
print("Are there any NaN or empty values in the column?",path.isnull().values.any(), "\nThere are: ",
path.isnull().values.sum(), "in the column")

# %% [markdown]
# There are no empty values in Pathrise_status column.

# %%
path = path.replace([["Withdrawn", "Withdrawn (Trial)", "MIA", "Deferred"]],"not_enrolled")
path = path.replace([["Placed", "Withdrawn (Failed)", "Active", "Closed Lost", "Break"]],"enrolled")

path.value_counts()

# %% [markdown]
# #### Quick visual: histogram of the enrolled vs not enrolled individuals

# %%
plt.hist(path, bins=2, histtype='step')
# Label the axes
plt.xlabel('Enrollment Status')
plt.ylabel('Count')
plt.title('Pathrise Enrollment Status')
# Show the figure
plt.show()

# %% [markdown]
# ##### Now import the changes made to both pathrise status and placed back into the main dataframe

# %%
df["placed"] = placed
df["pathrise_status"] = path


# %% [markdown]
# ### Clean Data, Missing Values

# %%
print("Are there any NaN or empty values in the dateframe?", df.isnull().values.any(), "\n",
df.isnull().sum() >0)


# %% [markdown]
# ##### Empty values
# In the dataframe, all but 5 columns contain empty/NaN values. 

# %%
print("There are: ", df["cohort_tag"].isnull().values.sum(), " in the cohort_tag column",
      "\nThere are: ", df["program_duration_days"].isnull().values.sum(), "in the program_duration_days column",
      "\nThere are: ", df["employment_status "].isnull().values.sum(), "in the employment_status column",
      "\nThere are: ", df["highest_level_of_education"].isnull().values.sum(), "in the highest_level_of_education column",
      "\nThere are: ", df["length_of_job_search"].isnull().values.sum(), "in the length_of_job_search column",
      "\nThere are: ", df["biggest_challenge_in_search"].isnull().values.sum(), "in the column",
      "\nThere are: ", df["professional_experience"].isnull().values.sum(), "in the professional_experience column",
      "\nThere are: ", df["work_authorization_status"].isnull().values.sum(), "in the work_authorization_status column",
      "\nThere are: ", df["number_of_interviews"].isnull().values.sum(), "in the number_of_interviews column",
      "\nThere are: ", df["gender"].isnull().values.sum(), "in the gender column",
      "\nThere are: ", df["race"].isnull().values.sum(), "in the race column"
     )

# %% [markdown]
# ##### Visual of missing values

# %%
msno.bar(df)
#plt.figure(figsize=(10,10))
#plt.xlabel('Columns', fontsize = 30 )
#plt.ylabel('Values')
plt.title('Missing Values', fontsize = 30)
# Show the figure
plt.show()


# %% [markdown]
# Reducing the the values into 2 different values rather than 9 will give us a better understanding of how successful the program is for the people that acutally participated in the program. There were 1784 participates that were involved in the program 

# %% [markdown]
# ### 

# %% [markdown]
# # Education impact, did individuals who had a formal education or higher education placed at a higher rate compared to other individuals? 

# %% [markdown]
# ### Part 1: What is the education of the fellows that were placed 

# %%
#First rename highest level of education to make it easier in the future
df = df.rename(columns={'highest_level_of_education': 'education', 'program_duration_days': "days_program"})

edu = df["education"]
edu_b = edu == "Bachelor's Degree"
edu_m = edu == "Master's Degree"
edu_scollege = edu == "Some College, No Degree"
edu_d = edu == "Doctorate or Professional Degree"
edu_hdrop = edu == "Some High School"
edu_hs = edu == "High School Graduate"
edu_ged = edu == "GED or equivalent"


# %%
edu_hs.value_counts()

# %%
edu_b_placement = edu_b[df["placed"]=="placed"]
edu_m_placement = edu_m[df["placed"]=="placed"]
edu_scollege_placement = edu_scollege[df["placed"]=="placed"]
edu_d_placement = edu_d[df["placed"]=="placed"]
edu_hdrop_placement = edu_hdrop[df["placed"]=="placed"]
edu_hs_placement = edu_hs[df["placed"]=="placed"]
edu_ged_placement = edu_ged[df["placed"]=="placed"]


print("There were",edu_b_placement.values.sum(), "fellows with Bachelor's Degrees that got placed",
     "\nThere were",edu_m_placement.values.sum(), "fellows with Master's Degrees that got placed",
      "\nThere were",edu_d_placement.values.sum(), "fellows with Doctorate or Professional Degree that got placed",
      "\nThere were",edu_scollege_placement.values.sum(), "fellows with Some College, No Degree that got placed",
      "\nThere were",edu_hdrop_placement.values.sum(), "fellows without a Highschool Degree that got placed",
      "\nThere were",edu_hs_placement.values.sum(), "fellows with a Highschool Degree that got placed",
      "\nThere were",edu_ged_placement.values.sum(), "fellows that recieved their highschool GED that got placed")


# %%
print((edu_b_placement.values.sum()/edu_b_placement.value_counts().sum())* 100, "%of the fellows who got placed had a Bachelor's Degree")
print((edu_m_placement.values.sum()/edu_m_placement.value_counts().sum())* 100, "%of the fellows who got placed had a Master's Degree")
print((edu_d_placement.values.sum()/edu_d_placement.value_counts().sum())* 100, "%of the fellows who got placed had a Doctorate or Professional Degree")
print((edu_scollege_placement.values.sum()/edu_scollege_placement.value_counts().sum())* 100, "%of the fellows who got placed had a some college education but did not recieve a degree")
print((edu_hdrop_placement.values.sum()/edu_hdrop_placement.value_counts().sum())* 100, "%of the fellows who got placed did not recieve a highschool high school diploma")
print((edu_hs_placement.values.sum()/edu_hs_placement.value_counts().sum())* 100, "%of the fellows who got placed recieved a highschool diploma")
print((edu_ged_placement.values.sum()/edu_ged_placement.value_counts().sum())* 100, "%of the fellows who got placed recieved their highschool GED")

      
      
      

# %% [markdown]
# ### Part 2: Education of fellows that were enrolled in the program, place and not placed and analysis 

# %% [markdown]
# Step 1: Remove the individuals who did not enroll in the program(withdrew) or went MIA
# NOTE: I did not remove empty values just yet, but I will be removing them shortly
# 

# %%
#Make a quick copy of the dataframe
df_copy = df.copy()
df_copy

# %%
#Removign all the 
df_copy = df_copy[df_copy.pathrise_status != "not_enrolled"]


# %%
df_copy
#There were a total of 1784 fellows who enrolled in the program.

# %%
#Check empty values before moving forward
print(df_copy['pathrise_status'].isnull().sum())
print(df_copy['placed'].isnull().sum())
print(df_copy['education'].isnull().sum())
df_copy1 = df.copy()
df_copy1["days_program"] =df_copy1[(df_copy1['days_program'] == 2)]
df_copy1["days_program"].value_counts()

# %% [markdown]
# ### Removing of missing values in Education
# Removing the missing values now may effect the duration analysis later on but it needs to be removed now to conduct the education and placement analysis.

# %%
#I will be removing the empty values from education
print(df_copy.shape)
df_copy.dropna(subset=['education'], inplace=True)
print(df_copy.shape[0])

# %%
##Bachelors
bach = df_copy.loc[(df_copy['education'] == "Bachelor's Degree")]
bach_placed = df_copy.loc[(df_copy['education'] == "Bachelor's Degree")& (df_copy['placed'] == "placed")] 
print(bach.education.value_counts().sum(), "of the fellows enrolled in the program have a Bachelor's Degree")
print(bach_placed.education.value_counts().sum(),"of the fellows who have a Bachelor's Degree got placed")
print(bach_placed.education.value_counts().sum()/bach.education.value_counts().sum() * 100,
      "% of fellows with a Bachelor's Degree got placed, the others are either active or failed from the program\n")        
##Masters 
masters = df_copy.loc[(df_copy['education'] == "Master's Degree")]
masters_placed = df_copy.loc[(df_copy['education'] == "Master's Degree")& (df_copy['placed'] == "placed")] 
print(masters.education.value_counts().sum(), "of the fellows enrolled in the program have a Master's Degree")
print(masters_placed.education.value_counts().sum(),"of the fellows who have a Master's Degree got placed")
print(masters_placed.education.value_counts().sum()/masters.education.value_counts().sum() * 100, 
      "% of fellows with a Master's Degree got placed, the others are either active or failed the program\n")          

##PHD
phd = df_copy.loc[(df_copy['education'] == "Doctorate or Professional Degree")]
phd_placed = df_copy.loc[(df_copy['education'] == "Doctorate or Professional Degree")& (df_copy['placed'] == "placed")] 
print(phd.education.value_counts().sum(), "of the fellows enrolled in the program have a Doctorate or Professional Degree")
print(phd_placed.education.value_counts().sum(),"of the fellows who have a Doctorate or Professional Degree got placed")
print(phd_placed.education.value_counts().sum()/phd.education.value_counts().sum() * 100, 
      "% of fellows with a Doctorate or Professional Degree got placed, the others are either active or failed the program\n")

##College_dropout
col_drop = df_copy.loc[(df_copy['education'] == "Some College, No Degree")]
col_drop_placed = df_copy.loc[(df_copy['education'] == "Some College, No Degree")& (df_copy['placed'] == "placed")] 
print(col_drop.education.value_counts().sum(), "of the fellows enrolled in the program have a Some College, but No Degree")
print(col_drop_placed.education.value_counts().sum(),"of the fellows who have a Some College, but No Degree got placed")
print(col_drop_placed.education.value_counts().sum()/col_drop.education.value_counts().sum() * 100, 
      "% of fellows with a Some College, No Degree got placed, the others are either active or failed the program\n")

##Highschool graduate
hs = df_copy.loc[(df_copy['education'] == "High School Graduate")]
hs_placed = df_copy.loc[(df_copy['education'] == "High School Graduate")& (df_copy['placed'] == "placed")] 
print(hs.education.value_counts().sum(), "of the fellows enrolled in the program have a High School Graduate")
print(hs_placed.education.value_counts().sum(),"of the fellows who have a High School Graduate got placed")
print(hs_placed.education.value_counts().sum()/hs.education.value_counts().sum() * 100, 
      "% of fellows with a High School Graduate got placed, the others are either active or failed the program\n")

##Highschool dropout
hs_drop = df_copy.loc[(df_copy['education'] == "Some High School")]
hs_drop_placed = df_copy.loc[(df_copy['education'] == "Some High School")& (df_copy['placed'] == "placed")] 
print(hs_drop.education.value_counts().sum(), "of the fellows enrolled in the program did not complete/recieve a High School diploma")
print(hs_drop_placed.education.value_counts().sum(),"of the fellows who did not recieve a high school diploma got placed")
print(hs_drop_placed.education.value_counts().sum()/hs_drop.education.value_counts().sum() * 100, 
      "% of fellows without a high school diploma got placed, the others are either active or failed the program\n")

##
ged = df_copy.loc[(df_copy['education'] == "GED or equivalent")]
ged_placed = df_copy.loc[(df_copy['education'] == "GED or equivalent")& (df_copy['placed'] == "placed")] 
print(ged.education.value_counts().sum(), "of the fellows enrolled in the program recieved their GED(or equivalent)")
print(ged_placed.education.value_counts().sum(),"of the fellows who recieved their GED(or equivalent) got placed")
print(ged_placed.education.value_counts().sum()/ged.education.value_counts().sum() * 100, 
      "% of fellows who recieved their GED (or equivalent) got placed, the others are either active or failed the program\n")



# %% [markdown]
# ### Analysis of Education and Placement
# 
# Throughout the research, we discovered a weak correlation between education and placement. Out of the 956 fellows that were placed, ~55% were fellows who had Bachelor's Degree, ~30% received their Masters, ~6% received their Doctorate or Professional Degree, ~5% had some college experience but did not receive a degree, ~.4% were fellows without a GED or a high school diploma, ~1.5% were fellows who received either their high school diploma or a GED certificate. We can conclude from these statistics is that the higher the education of the fellow, the more likely you will be placed. Once cleaning the education variable and removing all empty values we determine that ~54% of the fellows that received a Bachelor's Degree got placed, ~52% of the fellows who received a Master’s got placed, ~58% of fellows who received either a Doctorate Degree or Professional Degree were placed, ~55 of the fellows with some college, but no degree got placed, ~77% of the fellows with high school diploma got placed, ~50% of the fellows without a high school diploma got placed, and ~71% of the fellows who received their GED (or equivalent) got placed. The statistics presented above provide evidence that education level does not make an impact on placement, but it does have an impact on entering the program. The majority of the fellows either have a college degree or higher, which means the individuals with lower education criteria are likely to struggle to enter the program during the application process. Another possibility is that many individuals with lower education criteria simply do not apply for the program. What can be concluded is the relationship between education and placement is flawed, meaning education does not play a significant part in being placed but rather plays an important part in entering Pathrise.    
# 

# %%
placed = df["placed"]

placed= placed.replace("not_placed",0)
placed = placed.replace("placed",1)

df["placed"] = placed
df["placed"]

# %% [markdown]
# # Duration until placement, how long does it take a fellow to be placed

# %%
#review feature
df_copy["days_program"]

# %%
#Review empty values
df_copy["days_program"].isnull().sum()


# %% [markdown]
# The feature contains 550 NaN/empty values, this is not entirely unexpected becauase many of the fellows included in the data are still at the status of "active", which means they have not been placed yet. The next steps are to remove the empty values from feature to continue the duration analysis. 

# %%
print(df_copy.shape[0])
df_copy.dropna(subset=['days_program'], inplace=True)
print(df_copy.shape[0])

# %%
df_copy["days_program"]


# %%
days_placed = df_copy.loc[(df_copy['placed'] == "placed")]
np.mean(days_placed["days_program"])

# %% [markdown]
# The average days in the program prior to being placed is around 161 days.

# %%
#Describtion of the feature
df_copy["days_program"].describe()

# %%
#Plot the distribution of the feature, days_program

plt.figure(figsize=(12,10))
sns.distplot(df_copy["days_program"])
# Label the axes
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.title('Days In The Program')
# Show the figure
plt.show()


# %% [markdown]
# The feature is distributed unevenly and the data is skewed to the right. A red flag is the large amount of fellows who were a part of the program for less than a day. Now that we knoew it is skewed to the right we can use the median, which is less than the mean. Another good thing is that the distribution is unimodal, meaning the feature contains a single highest value.

# %% [markdown]
# # log transformation to normalize data
# I decided that I will keep the empty values out of the data moving forward.

# %%
def normal(data):
    """
    Returns True if the data given is normal ie. the p-value is less than 0.05. 
    If not, then return False.
    
    Parameters:
    data - The data to test normality of.
    
    Returns True if the data is normal, False otherwise.
    """

    _, p = stats.normaltest(data, nan_policy = 'omit')
    if p < 0.05:
        return False
    else:
        return True

# %%
if normal(df['days_program']):
    print('The data is normal!')
else:
    print('It is not normal!!!!!!!')

normal(df['days_program'])

# %%
print('Skewness:', df['days_program'].skew())
print('Kurtosis:', df['days_program'].kurtosis())

# %% [markdown]
# ##### Skewness
# The skewness is .814, and the Kurtosis is .0049.
# A skewness of .814 means the data is moderately skewed, a kurtosis(describes the shape)  of .0049 is less than 3 which means it is platykurtic. The distribution is not normal.

# %%
prob_plot = stats.probplot(df['days_program'], plot = plt)

# %%
#Find the median because it is not normalized
df['days_program'].median()
#Median == 112

# %% [markdown]
# #### Key take aways froms skewness tests and distribution
# 1. The median, skewness/Kurtosis value, and histogram of the distribution all indicate that the data is unimodal distribution skewed to the right.
# 2. Biggest issue is that many fellows did not particapte in the program(0 days in the program) and it has been included in the analysis. 
# 3. The data is NOT normal and we need to normalize the data. 
# 

# %% [markdown]
# # Imputing the data through linear regression modeling
# **The decision to impute the data rather than remove the missing is mainly because the data file is small. I want to include as many data values as possible to increase the effectiveness of our two models. 
# 1. imputation through linear regression 

# %%
#Break the data into categorical and quantitative features 
cat_feat = ['gender', 'race', 'pathrise_status', 'employment_status ', 
              'education', 'length_of_job_search', 
              'professional_experience', 'work_authorization_status', 
              'primary_track', 'cohort_tag', 'placed']
quant_feat = ['number_of_interviews', 'number_of_applications']

# %%
# One-hot encode cat_feat
df_impute = pd.get_dummies(df[cat_feat])

# %%
#bring in quantitative features
df_impute[quant_feat] = df[quant_feat]

# %%
#impute number of interviews using the median
df_impute['number_of_interviews'].fillna(df_impute['number_of_interviews'].median(), inplace = True)

# %%
df_impute['days_program'] = df['days_program']

# %%
#The linear regression model, train the data on the days_program
#that is not missing days_program -- The model will predict the 
#days_program missing values. (notnull = nonmissing values, isna = missing values)

training_set = df_impute[df_impute['days_program'].notnull()]
testing_set = df_impute[df_impute['days_program'].isna()]

X_train = training_set.iloc[:, 0:-1]
y_train = training_set['days_program']

# Train the model
lin_reg = LinearRegression().fit(X_train.values, y_train.values)

# Check out first 10 predictions
X_test = testing_set.iloc[:, 0:-1]
lin_reg.predict(X_test)[0:10]
#These values are extreme and are not consistent with the other values in the feature

# %%
#I will try again using the Lasso method
clf = linear_model.Lasso(alpha = 0.1)
clf.fit(X_train.values, y_train.values)

# Check out first 10 predictions
clf.predict(X_test)[0:10]

#These values are more consistent and this is exactly what we needed

# %%
df.loc[testing_set.index.values, "days_program"] = clf.predict(X_test)

# %%
df.isna().sum()

# %% [markdown]
# ### Now it is time to clean up the data a little more, focusing solely on the na values
# 1. either drop na if there are not many values
# 2. use fillna to fill in the missing values
# 3. possibly imputing using the median() value

# %%
#drop na values for these three features because there are not many empty values
df.dropna(subset = ["work_authorization_status", 'employment_status ','education', 'length_of_job_search','gender', 'race', 'cohort_tag', 'biggest_challenge_in_search',"number_of_interviews",], inplace = True)


# %%
#professional_experience
df['employment_status '].fillna('MISSING_EMPL_STATUS', inplace = True)

# %%

df.isna().sum()

# %% [markdown]
# ### Simple log transformation (continued)
# 

# %%
#Simply log transformation
df["log_days"] = df["days_program"].apply(lambda x: np.log(abs(x)) if x > 0 else 0)


# %%
df["log_days"]

# %%
plt.figure(figsize=(12,10))
sns.distplot(df["log_days"])
# Label the axes
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.title('View the change in distribution(is it normalized now?)')
# Show the figure
plt.show()

# %%
df

# %% [markdown]
# # Modeling
# Reviewing the problem again, it is critical to clearly state the problem. 

# %% [markdown]
# ### Problem Overview
# 
# We will be utilizing two different models
# 
# #1. Regression model: To determine WHEN a fellow will be placed. Due to deadline requirements, I will only conduct a classification model. I will continue to work on the regression model in the mean time.  
# 
# #2. Classification model: to determine IF a fellow will be placed.
# 
# 
# The two dependant variables are clear, for classification the dependent variable is placed, and for the regression model the dependent variable is days_program.

# %% [markdown]
# ### One Hot encoding, independent variables

# %% [markdown]
# # Categoical Variables

# %%
df

# %%
cat_feat = [
    "primary_track", "cohort_tag", 
    "employment_status ", "education", "length_of_job_search",
    "professional_experience", "work_authorization_status","gender","race"
]

# I will not include the feature pathrise_status, because it contains only one value:enrolled. 
#All the individuals in the df_copy dataframe are fellows and they particpated in the program.
#I thought about removing race and gender features but decided that they could provide some useful insights on the data.
    
    
    

# %%
#Create a one-hot encoding function for classification categorical variables

one_hot_classification = []
one_hot_regression = []

def one_hot(variable, for_placed = True):
    for u_variable in df[variable].unique():
        if for_placed: 
            one_hot_classification.append(u_variable)
        else:
            one_hot_regression.append(u_variable)
    
    df_new_dum = pd.get_dummies(df[variable])    
    df[df_new_dum.columns] = df_new_dum


# %% [markdown]
# # Independent Variable Analysis
# This needs to be done prior to modeling/hot encoding

# %% [markdown]
# ### Independent variable 1, Primary Track

# %%
plt.figure(figsize=(12,10))
df.groupby("primary_track").size().plot(kind='bar')
# Label the axes
plt.xlabel("Track")
plt.ylabel("Frequency")
plt.title("Primary Track Distribution")
# Show the figure
plt.show()

# %% [markdown]
# ### Check missing values of primary track

# %%
print("Are there any NaN or empty values in the column?", df_copy["primary_track"].isnull().values.any(), "\nThere are: ",
df["primary_track"].isnull().values.sum(), "in the column")


# %% [markdown]
# The primary track feature, describes the career path of the fellow. There are six different career paths, data science, design, marketing, PSO, SWE, and Web. In the image we can clearly see that majority of the enrolled fellows are Software Engineers (SWE). Another important thing to note is that there are no missing values for the primary track feature.  

# %% [markdown]
# ### Independent variable 2, Cohort Tag
# 

# %%
plt.figure(figsize=(12,10))
df.groupby("cohort_tag").size().plot(kind='bar')
# Label the axes
plt.xlabel("Cohort")
plt.ylabel("Frequency")
plt.title("Cohort Distribution")
# Show the figure
plt.show()

# %%
print(df["cohort_tag"].value_counts())
#There is an issue with Cohort in the data, there seems to be lowercases in some of the cohorts,
#and I assume these are typos.

df_lower = df["cohort_tag"] == "FEB20a"
df_upper = df["cohort_tag"] == "FEB20A"

print("There were",df_lower.values.sum(), "fellows with in Cohort FEB20a")
print("There were",df_upper.values.sum(), "fellows with in Cohort FEB20A")

#Quick fix

df["cohort_tag"] = df["cohort_tag"].replace("FEB20a","FEB20A")
df_lower2 = df["cohort_tag"] == "FEB20a"
print("There were",df_lower2.values.sum(), "fellows with in Cohort FEB20a")

#All I did was replace FEB20a with FEB20A
#Another issue, OCT21A-- a future day possibly a typo, but because there is no OCT 20 I do
#not want to take the risk of including this data with OCT19 and decided it is best for the data
#to be removed from the data.
print(df.shape)
df.drop(df[df["cohort_tag"] == "OCT21A"].index, inplace = True)
print(df.shape)


# %% [markdown]
# ### Independent variable 3, Employment Status
# 

# %%
plt.figure(figsize=(12,10))
df.groupby("employment_status ").size().plot(kind='bar')
# Label the axes
plt.xlabel("Status")
plt.ylabel("Frequency")
plt.title("Employment Status of Fellows Prior to Pathrise")
# Show the figure
plt.show()

# %% [markdown]
# ### Independent variable 4, Education
# 

# %% [markdown]
# We determine earlier that education does not play a big part in placement, rather it plays a bigger part in being included in the program. 

# %%
plt.figure(figsize=(12,10))
df.groupby("education").size().plot(kind='bar')
# Label the axes
plt.xlabel("Education")
plt.ylabel("Frequency")
plt.title("Highest Level of Education of a Fellow")
# Show the figure
plt.show()

# %% [markdown]
# The feature, education, describes the highest level of education a fellow has received. We can see in the visual of the distribution, that majority of the fellows in the program received either a Bachelor’s Degree or a Master’s Degree. In our earlier analysis of placement and education we determine that education level did not affect placement, but the data suggest that fellows with higher education are more likely to successful joining the program. Prior to the analysis I assumed the higher the education level of a fellow the more likely a fellow will be placed, but the data makes it evident that there is no strong relationship between placement and education. 

# %% [markdown]
# ### Independent variable 5, Length of Job Search
# 

# %%
plt.figure(figsize=(12,10))
df.groupby("length_of_job_search").size().plot(kind='bar')
# Label the axes
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.title("The length of a fellows Job Search")
# Show the figure
plt.show()

# %% [markdown]
# ### Independent variable 6, Work Authorization Status
# 

# %%
plt.figure(figsize=(12,10))
df.groupby("work_authorization_status").size().plot(kind='bar')
# Label the axes
plt.xlabel("Status ")
plt.ylabel("Frequency")
plt.title("Fellows Work Authorization Status")
# Show the figure
plt.show()

# %% [markdown]
# Many of the fellows are American Citizens or have the F1 Visa/OPT. Work Authorization is a big factor in getting placed with company due to many companies seeking individuals with authorization to work in the United States. It is a rather huge obstacle for a company to sponser an applicant. 

# %% [markdown]
# ### Independent variable 7, Professional Experience
# 

# %%
plt.figure(figsize=(12,10))
df.groupby("professional_experience").size().plot(kind='bar')
# Label the axes
plt.xlabel("Years of Experience")
plt.ylabel("Frequency")
plt.title("Work Experience of Pathrise Fellows")
# Show the figure
plt.show()

# %% [markdown]
# ### Independent variable 8, Race
# 

# %%
plt.figure(figsize=(12,10))
df.groupby("race").size().plot(kind='bar')
# Label the axes
plt.xlabel("Race")
plt.ylabel("Frequency")
plt.title("Race Distribution of Fellows")
# Show the figure
plt.show()

# %%
#The visual distribution above does not include the missing values. 
plt.figure(figsize=(12,10))
df.groupby("race").size().plot(kind='bar')
# Label the axes
plt.xlabel("Race")
plt.ylabel("Frequency")
plt.title("Race Distribution of Fellows")
# Show the figure
plt.show()

# %% [markdown]
# ### Independent variable 9, Gender
# 

# %%
plt.figure(figsize=(12,10))
df.groupby("gender").size().plot(kind='bar')
# Label the axes
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.title("Gender Distribution")
# Show the figure
plt.show()

# %%
#The visual distribution above does not include the missing values. 
plt.figure(figsize=(12,10))
df.groupby("gender").size().plot(kind='bar')
# Label the axes
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.title("Gender Distribution including Missing Values")
# Show the figure
plt.show()

# %% [markdown]
# # Chi-Square Test

# %% [markdown]
# In order to determine if the categorical variables and the dependent variables have a significant relationship we need to run a Chi-square test. If the p-value is less than .05 then we can reject the null value and there is a significant relationship between the dependent variable and the categorical variable. Chi-square test is discussed in more depth in the research paper.

# %%
sig_classification = [] 
for cat in cat_feat:    
    cross_table = pd.crosstab((df[cat]), df['placed'],margins = False)
    stat, p, dof, expected = stats.chi2_contingency(cross_table.values)
    if p < 0.05:
        sig_classification.append(cat)

# %%
print("These categorical independent variables:",  sig_classification,"interact with placed, the dependent variable")
df

# %%


# %% [markdown]
# ### One Hot encode 

# %%
one_hot_classification

# %%
for i in sig_classification:
    one_hot(i, True)


# %% [markdown]
# # Which categorical variables have a significant relationship with days_program (regression problem)

# %%
def determine_sig_anova(cat, dv):
    """
    Determines the significance of a categorical variable using an ANOVA test.
    
    Parameters:
    categ_var - The feature to test on.
    dv - The feature to test against.
    
    Returns True if the groups are different, False otherwise
    """
    levels = df[cat].unique()
    all_types = [] # List of lists for each level's program_duration_days
    for level in levels:
        prog_dur_days = df[df[cat] == level][dv].values
        all_types.append(prog_dur_days)
    
    # Reject H0?
    pval = stats.f_oneway(*all_types)[1]
    if pval < 0.05:
        return True
    else:
        return False

# %%
timer = [] # Holds statistically significant variables for regression problem

for cat in cat_feat:
    if determine_sig_anova(cat, 'days_program'):
        timer.append(cat)

# %%
timer 

# %%
for time_1 in timer:
    one_hot(time_1, False)

# %%
time_1

# %%


# %% [markdown]
# ### The signifcant categorical features for the models
# #### Classification Model:
# 1. primary_track
# 2. cohort_tag
# 3. gender
# 4. race
# 
# #### Regression Model:
# 1. cohort_tag
# 2. work_authorization_status
# 3. gender

# %% [markdown]
# # Quantitative variables

# %%
#These are the three quantitative variables I am going to focus on for the classification model and the regression model
print(df["number_of_interviews"])
print(df["number_of_applications"])
print(df["professional_experience"])



# %% [markdown]
# ### Analyze the distribution of the quantitative variables

# %%
#The visual distribution above does not include the missing values. 
plt.figure(figsize=(12,10))
df.groupby("number_of_interviews").size().plot(kind='bar')
# Label the axes
plt.xlabel("Interviews")
plt.ylabel("Frequency")
plt.title("The Distribution of Fellow Interviews")
# Show the figure
plt.show()

# %%
#The visual distribution above does not include the missing values. 
plt.figure(figsize=(12,10))
df.groupby("number_of_applications").size().plot(kind='bar')
# Label the axes
plt.xlabel("Applications")
plt.ylabel("Frequency")
plt.title("The Distribution of Fellow Applications")
# Show the figure
plt.show()

# %% [markdown]
# ### Convert Professional experience into a quantitative variable
# Feature Engineering

# %%
print(df["professional_experience"].value_counts())

# %% [markdown]
# The idea will be to convert the value, "less than one year" to 6 representing months, 1-2 years converting it to 18 months (1.5 years), 3-4 years to 42 months and 5+months to 60 months.
# 

# %%
df["professional_experience"] = df["professional_experience"].replace(["Less than one year"], 6)
df["professional_experience"] = df["professional_experience"].replace(["1-2 years"], 18)
df["professional_experience"] = df["professional_experience"].replace(["3-4 years"], 42)
df["professional_experience"] = df["professional_experience"].replace(["5+ years"], 60)
df["professional_experience"].value_counts()


# %%
#These are for the classification problem (PLACEMENT)
quan = ["days_program","number_of_interviews", "number_of_applications", "professional_experience"]
quan_hot =[]

# %%
quan

# %%
quan_placed = []

# %%
for quan_1 in quan:
    if determine_sig_anova('placed', quan_1):
        quan_placed.append(quan_1)


# %%
def anova(cat, var):
    ls = df[cat].unique()
    listt = []
    for l in ls:
        days = df[df[cat] == l][var].values
        listt.append(days)
    p_value = stats.f_oneway(*listt)[1]
    if p_value < 0.05:
        return True
    else:
        return False

# %%
stat_quan = []

for q in quan:
    if anova('placed', q):
        stat_quan.append(q)

# %%
print("These quantitative independent variables:",  stat_quan ,"interact with placed, the dependent variable")

# %% [markdown]
# ### Quantitative overview
# 
# I thought there would be more than just one quantitative variable that interacted with placed. I am not surprised that days_program was the variable(days_program is independent variable in the classification model, but dependent in the regression model). The amount of interviews a fellow has recieved or the amount of applications that fellow submitted does not have a signficant impact on placement, which is odd in my opinion. 

# %%
# Get a quick look at the correlations between the quantitative variables and placement
corrmat = df_copy[["days_program", 'number_of_interviews', 'number_of_applications']].corr()

f, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Heatmap of Quantitative Variables vs Days in Program')
k = 10 
cols = corrmat.nlargest(k, 'days_program')['days_program'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, 
                 square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# %% [markdown]
# #### Heat Map Analysis
# So there is nothing that really stands out in the heatmap. Unfortunately I will exclude all quantitive features from the regression model, no value is greater than .10 (Closer to 1 the more significant the relationship). Currently the number of applications and number of interviews both are -.06 and -.08 respectively. 

# %% [markdown]
# # Modeling -- Classification Model

# %%
# Isolate the features for both the classifcation models 
placed_features = one_hot_classification + quan_hot 
placed_features.append("log_days")

placement_time_features = one_hot_regression + quan_hot

# Split into training and test sets for both the regression and classification models
placed_X_train_val, placed_X_test, placed_y_train_val, placed_y_test = train_test_split(df[placed_features].values, 
                                                                                        df['placed'].values, 
                                                                                        test_size = 0.2, 
                                                                                        random_state = 42)

placement_time_X_train_val, placement_time_X_test, placement_time_y_train_val, placement_time_y_test = train_test_split(df[placement_time_features].values, 
                                                                                                                    df['log_days'].values, 
                                                                                                                        test_size = 0.2, random_state = 42)

#number of folds
n_folds = 30

# %%


# %% [markdown]
# # Placement Time
# #### It is important to note that I will be using F1 score as a metric to identify the models effectiveness becuase the data set is imbalanced. Accuracy would not be a good metric to assess the effectiveness of the model. 

# %% [markdown]
# # Logistic Model: Classification Model #1 

# %%
kf = KFold(n_splits = n_folds, shuffle = True, random_state = 42)

train_f1_scores = []
val_f1_scores = []
placed_lreg = LogisticRegression()

# Perform K-Fold CV
for train_inds, val_inds in kf.split(placed_X_train_val):
    X_train, X_val = placed_X_train_val[train_inds], placed_X_train_val[val_inds]
    y_train, y_val = placed_y_train_val[train_inds], placed_y_train_val[val_inds]
    
    placed_lreg.fit(X_train, y_train)
    train_preds = placed_lreg.predict(X_train)
    val_preds = placed_lreg.predict(X_val)
    
    train_f1_scores.append(f1_score(y_train, train_preds))
    val_f1_scores.append(f1_score(y_val, val_preds))

# %%
f, ax = plt.subplots(figsize=(10,8))
plt.plot(np.arange(0, n_folds), val_f1_scores, label = 'Validation F1 Score')
plt.plot(np.arange(0, n_folds), train_f1_scores, label = 'Training F1 Score')
plt.ylabel('F1 Score')
plt.title('Simple Logistic Regression F1 Score over 30-Fold CV')
plt.legend()
plt.show()

# %%
print('Best Validation F1 Score:', val_f1_scores[np.argmax(val_f1_scores)])
# See how well it performs on the test set
print('F1 Score on Test Set: %.4f' % f1_score(placed_y_test, placed_lreg.predict(placed_X_test)))

# %% [markdown]
# ### Quick Logistic Regression Model Evulation
# The model yielded an F1 score of .71 (best validation) and F1 Score of .51 on the test Set. The two scores are actually good and they both indicate that the model is effective.

# %% [markdown]
# ### Compute ROC curve and AUC 
# 

# %%
fpr, tpr, thresholds = roc_curve(placed_y_test, 
                                 placed_lreg.decision_function(placed_X_test))
calc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange',
         lw = 2, label='ROC curve (area = %0.2f)' % calc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### Analysis on the ROC Curve and AUC 
# I get a AUC of .74 which is considered acceptable and the logistic regression model is a decent one. Although it is not a high .8 or even a .9 it is still acceptable and the model is effectively predicts if a pathrise fellow will be placed. 

# %% [markdown]
# ### XGBOOST MODEL

# %%
xgb_fit_dict = {
    'eval_metric': 'auc',
    "early_stopping_rounds": 15,
    "eval_set": [(placed_X_test, placed_y_test)],
    'verbose': 100
}

xgb_param_dict = {
    'n_estimators': np.arange(10, 100, 10),
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'learning_rate': [0.05, 0.1, 0.3],
    'subsample': stats.uniform(loc=0.2, scale=0.8),
    'colsample_bytree': stats.uniform(loc=0.4, scale=0.6),
    'gamma': [0.0, 0.1, 0.2],
    'max_depth': [5, 7, 10],
    'min_child_samples': stats.randint(100, 500), 
    "objective": ["binary:logistic"],
    'alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
}


placed_xgb_model = XGBClassifier(seed = 42, metric = 'None', n_jobs = 4, silent = True)

rs_clf = RandomizedSearchCV(placed_xgb_model, random_state = 42, 
                            param_distributions = xgb_param_dict, 
                            n_iter = 50, 
                            cv = n_folds, 
                            scoring = 'f1', 
                            verbose = False) 

# %%
from os import path
# Only train if there does not exist a saved model
placed_saved = 'placed_best.pickle.dat'

if path.exists(placed_saved):
    print('I already have a saved model.')
    
    # Load in saved model
    placed_best = pickle.load(open(placed_saved, 'rb'))
    
    print('Saved Model Parameters')
    print(placed_best.get_xgb_params())
    
    # Compute saved model's MSE for test set
    best_xgb_preds = placed_best.predict(placed_X_test)
    start = time.time()
    print("Model took %.2f seconds to complete." % (time.time()-start))
    print("F1 Score on Test Set: %.4f" % f1_score(placed_y_test, best_xgb_preds))
    
else:
    print('Starting to train...')
    
    # Fit via RandomizedSearch
    start = time.time()
    rs_clf.fit(placed_X_train_val, placed_y_train_val, **xgb_fit_dict)
    print("RandomizedSearch took %.2f seconds to complete." % (time.time()-start))
    
    # Get best params
    xgb_best_params = rs_clf.best_params_
    
    # Train using best params
    placed_best = XGBClassifier(**xgb_best_params, seed = 42)
    start = time.time()
    placed_best.fit(placed_X_train_val, placed_y_train_val)
    
    # Get MSE
    best_xgb_preds = placed_best.predict(placed_X_test)
    print("Model took %.2f seconds to complete." % (time.time()-start))
    print("F1 Score on Test Set: %.4f" % f1_score(placed_y_test, best_xgb_preds))
    
    # Save best xgb model
    pickle.dump(placed_best, open(placed_saved, 'wb'))

# %%
# Compute ROC curve and AUC 
fpr, tpr, thresholds = roc_curve(placed_y_test, 
                                 placed_best.predict(placed_X_test))
calc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange',
         lw = 2, label='ROC curve (area = %0.2f)' % calc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### XGBoost Model Results and Analysis
# The XGBoost Model performs well but not as well as simple logistic regression model that I ran prior. They yield similar results but the logistic regression model is slightly a better model. 

# %%
f, ax = plt.subplots(figsize=(15,20))
xgb.plot_importance(placed_best, ax = ax)
plt.show()

# %%
# What are the some of the most important features?
print(df[placed_features].columns[62])
print(df[placed_features].columns[0])
print(df[placed_features].columns[50])
print(df[placed_features].columns[3])
print(df[placed_features].columns[60])
print(df[placed_features].columns[54])
print(df[placed_features].columns[51])
print(df[placed_features].columns[1])
print(df[placed_features].columns[21])
print(df[placed_features].columns[55])
print(df[placed_features].columns[58])

# %% [markdown]
# ### Important features for placement modeling
# * The log_days(days a fellow is in the program) turned out to be the most important feature in predicting placement, but I honestly figured that it would have a significant impact on placement.
# 
# * Another one that caught my attention was SWE was the second most important feature in placement. Pathrise seems to be the best a placing SWE compared to others. 
# 
# * The gender of the fellow as well as the fellows' ethnicity had a significant impact on placement.  

# %% [markdown]
# # Predicting Days Until Placement

# %% [markdown]
# ### We are going to start with a simple linear regression

# %%
kf = KFold(n_splits = n_folds, shuffle = True, random_state = 42)

train_mses = []
val_mses = []
placement_time_lreg = LinearRegression()

# Perform K-Fold CV
for train_inds, val_inds in kf.split(placement_time_X_train_val):
    X_train, X_val = placement_time_X_train_val[train_inds], placement_time_X_train_val[val_inds]
    y_train, y_val = placement_time_y_train_val[train_inds], placement_time_y_train_val[val_inds]
    
    placement_time_lreg.fit(X_train, y_train)
    train_preds = placement_time_lreg.predict(X_train)
    val_preds = placement_time_lreg.predict(X_val)
    
    train_mses.append(mean_squared_error(y_train, train_preds))
    val_mses.append(mean_squared_error(y_val, val_preds))

# %%
f, ax = plt.subplots(figsize=(10,8))
plt.plot(np.arange(0, n_folds), val_mses, label = 'Validation MSE')
plt.plot(np.arange(0, n_folds), train_mses, label = 'Training MSE')
plt.ylabel('MSE')
plt.title('Simple Linear Regression MSE over 10-Fold CV')
plt.legend()
plt.show()

# %%
print('Best Validation MSE:', val_mses[np.argmax(val_mses)])

# %%
print('MSE on Test Set: %.4f' % mean_squared_error(placement_time_lreg.predict(placement_time_X_test), placement_time_y_test))

# %%


# %% [markdown]
# ### XGBoost on Days Until Placement 

# %%
xgb_fit_dict = {
    'eval_metric': 'rmse',
    "early_stopping_rounds": 15,
    "eval_set": [(placement_time_X_test, placement_time_y_test)],
    'verbose': 100
}

xgb_param_dict = {
    'n_estimators': np.arange(10, 100, 10),
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'learning_rate': [0.05, 0.1, 0.3],
    'subsample': stats.uniform(loc=0.2, scale=0.8),
    'colsample_bytree': stats.uniform(loc=0.4, scale=0.6),
    'gamma': [0.0, 0.1, 0.2],
    'max_depth': [5, 7, 10],
    'min_child_samples': stats.randint(100, 500), 
    "objective": ["reg:squarederror"],
    'alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
}

placement_time_xgb_model = XGBRegressor(seed = 42, metric = 'None', n_jobs = 4, silent = True)

rs_clf = RandomizedSearchCV(placement_time_xgb_model, random_state = 42, 
                            param_distributions = xgb_param_dict, 
                            n_iter = 50, 
                            cv = n_folds, 
                            scoring = 'neg_mean_squared_error', 
                            verbose = False) 

# %%
placement_saved_model = 'placement_best_model.pickle.dat'

if path.exists(placement_saved_model):
    print('I already have a saved model.')
    
    # Load in saved model
    placement_best_model = pickle.load(open(placement_saved_model, 'rb'))
    
    print('Saved Model Parameters')
    print(placement_best_model.get_xgb_params())
    
    # Compute saved model's MSE for test set
    best_xgb_preds = placement_best_model.predict(placement_time_X_test)
    start = time.time()
    print("Model took %.2f seconds to complete." % (time.time()-start))
    print("MSE on Test Set: %.4f" % mean_squared_error(placement_time_y_test, best_xgb_preds))
    
else:
    print('Starting to train...')
    
    # Fit via RandomizedSearch
    start = time.time()
    rs_clf.fit(placement_time_X_train_val, placement_time_y_train_val, **xgb_fit_dict)
    print("RandomizedSearch took %.2f seconds to complete." % (time.time()-start))
    
    # Get best params
    xgb_best_params = rs_clf.best_params_
    
    # Train using best params
    placement_best_model = XGBRegressor(**xgb_best_params, seed = 42)
    start = time.time()
    placement_best_model.fit(placement_time_X_train_val, placement_time_y_train_val)
    
    # Get MSE
    best_xgb_preds = placement_best_model.predict(placement_time_X_test)
    print("Model took %.2f seconds to complete." % (time.time()-start))
    print("MSE on Test Set: %.4f" % mean_squared_error(placement_time_y_test, best_xgb_preds))
    
    # Save best xgb model
    pickle.dump(placement_best_model, open(placement_saved_model, 'wb'))


# %%
f, ax = plt.subplots(figsize=(15,20))
xgb.plot_importance(placement_best_model, ax = ax)
plt.show()

# %%
# What are the some of the most important features?
print(df[placement_time_features].columns[54])
print(df[placement_time_features].columns[1])
print(df[placement_time_features].columns[47])
print(df[placement_time_features].columns[2])
print(df[placement_time_features].columns[53])
print(df[placement_time_features].columns[48])
print(df[placement_time_features].columns[8])
print(df[placement_time_features].columns[39])
print(df[placement_time_features].columns[4])
print(df[placement_time_features].columns[50])

# %% [markdown]
# ### Review of XGBoost Model 
# The XGBoost model yielded an MSE of 2.7 MSE, which is the Mean Squared Error -- the difference between predicted and the actual values. The XGBoost regession model showed to be less effective than the linear regression model. 
# 
# Both these models do not effectively predict the placement time of the fellows and further research is required to enhance the models or even creating a different model will prove to be worth while. The research did provide some key insights that I will provide below. 
# 
# #### Features
# The important features comes as a huge surprise to me because placed is not a top feature/significant feature. I would assume that placed would have a more significant impact than Male (gender) which turned out to be the most significant feature. The JAN20A (cohort) is feature that I didn't think matter much but to be fair Janrary is the best time to apply for jobs so it makes sense. Corhort seems to be a top feature as it has a few values that are important to placement time. 

# %% [markdown]
# # Conclusion
# 1. ### Classification model 
#       The Classification model yielded the best results. The logistic regression with an AUC of .74 **(XGBoost had an AUC .70)**. Although it is not perfect, it is still deemed as an effective placement classifier. There are still a few things that can be done to further enhance the model, such as implementing a few feature engineering techniques. the insights of the model are provided below.
#     * The categorical features with significance for classification model:
#         1. primary_track
#         2. cohort_tag
#         3. gender
#         4. race
# 2. ### Regression Model
# The Linear Regression model was not a great one, yielding an MSE of 2.6 and 2.7(XGBoost model). This is a log days so the inverse log of 2.6 is 398 days, which is a huge error. Although the creation of the model did not provide effective predictions on placement time, it still gave us a lot of beneficial insights that could be beneficial for future advancements.
#     * The categorical features with significance for Linear Regression Model:
#     1. cohort_tag
#     2. work_authorization_status
#     3. gender
# 
# 
# ## Key insights of the research
# 
# 1. * Education did not have a big impact on placement. It did not matter what type of education you had to get placed, getting a less than an HS diploma might've had an impact on being able to enter to program but did not have an impact on placement. The model provided even more evidence that the education did not impact placement, education was not in the top 10 significant features in the model. This allows us to conclude that:
#     * Higher education did not mean higher placement rate! 
# 2. * Many of the fellows had a less than 2 years of experience(many were college students)
# 3. * Majority of the fellows were legally allowed to work in the United States. 
# 4. * ON average each fellow sent out an average of 20 applications. 
# 5. * Many of the fellows who applied to pathrise for were struggling to hear back from recruiters. 
# 6. * Unemployed and they were Male
# 7. * ### The median time a fellow stays in the program is around 111 days, that is around the amount that pathrise states. 
# 8. * ### Sending out many Applications is not correlated with getting placed faster and had a minimal impact on placement.

# %%



