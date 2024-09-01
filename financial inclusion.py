import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
# from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
plt.rcParams["axes.labelsize"] = 18
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

train = pd.read_csv('F:\\DATA SCIENCE SKILLS TECHNOLOGY\\FIELD PPT 2024- AI, ML\\financial-inclusion-in-africa\\Train.csv')
test = pd.read_csv('F:\\DATA SCIENCE SKILLS TECHNOLOGY\\FIELD PPT 2024- AI, ML\\financial-inclusion-in-africa\\Test.csv')
ss = pd.read_csv('F:\\DATA SCIENCE SKILLS TECHNOLOGY\\FIELD PPT 2024- AI, ML\\financial-inclusion-in-africa\\SampleSubmission.csv')
variables = pd.read_csv('F:\\DATA SCIENCE SKILLS TECHNOLOGY\\FIELD PPT 2024- AI, ML\\financial-inclusion-in-africa\\VariableDefinitions.csv')

# Let’s observe the shape of our datasets.
print('train data shape :', train.shape)
print('test data shape :', test.shape)

#show list of columns in train data
list(train.columns)

# inspect train data
train.head()

# Check for missing values
print('missing values:', train.isnull().sum())

# Explore Target distribution
sns.catplot(x="bank_account", kind="count",
            data=train, palette="Set1")

# view the submission file
ss.head()

#show some information about the dataset
print(train.info())

# Frequency table of a variable will give us the count of each category in that Target variable.
train['bank_account'].value_counts()

# Explore Target distribution

sns.catplot(x="bank_account", kind="count", data= train)

# Explore Country distribution

sns.catplot(x="country", kind="count",
            data=train, palette="colorblind")

# Explore Location distribution
sns.catplot(x="location_type", kind="count", data=train, palette="colorblind")

# Explore year distribution
sns.catplot(x="year", kind="count", data=train, palette="colorblind")

# Explore cellphone_access distribution
sns.catplot(x="cellphone_access", kind="count", data=train, palette="colorblind")

# Explore gender_of_respondent distribution
sns.catplot(x="gender_of_respondent", kind="count", data=train, palette="colorblind")

# Explore relationship_with_head distribution

sns.catplot(x="relationship_with_head", kind="count", data=train, palette="colorblind");

plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

# Explore marital_status distribution

sns.catplot(x="marital_status", kind="count", data=train, palette="colorblind");

plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

# Exp# Explore marital_status distribution

sns.catplot(x="marital_status", kind="count", data=train, palette="colorblind");

plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

sns.catplot(x="education_level", kind="count", data=train, palette="colorblind");

plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

# Explore job_type distribution

sns.catplot(x="job_type", kind="count", data=train, palette="colorblind");

plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

# Explore household_size distribution

plt.figure(figsize=(16, 6))
train.household_size.hist()
plt.xlabel('Household  size')

# Explore household_size distribution

plt.figure(figsize=(16, 6))
train.household_size.hist()
plt.xlabel('Household  size')

# Get the count of each unique household size
household_size_counts = train.household_size.value_counts()

# Find the most common household size and its count
most_common_household_size = household_size_counts.idxmax()
print("Most common number of people in household:", most_common_household_size)

most_common_household_size_count = household_size_counts.max()
print("Count of the most common household size:", most_common_household_size_count)

# Explore age_of_respondent distribution
plt.figure(figsize=(16, 6))
train.age_of_respondent.hist()
plt.xlabel('Age of Respondent')

plt.figure(figsize=(16, 6))
sns.catplot(x='location_type', hue='bank_account', 
            kind='count', data=train)
plt.xticks(
    fontweight='light',
    fontsize='x-large'
)

plt.figure(figsize=(16, 6))
sns.catplot(x='gender_of_respondent', hue='bank_account', kind='count', data=train)
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

plt.figure(figsize=(16, 6))
sns.catplot(x='cellphone_access', hue='bank_account', kind='count', data=train)
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

plt.figure(figsize=(16, 6))
sns.catplot(x='relationship_with_head', hue='bank_account', kind='count', data=train)
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

plt.figure(figsize=(16, 6))
sns.catplot(x='marital_status', hue='bank_account', kind='count', data=train)
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

plt.figure(figsize=(16, 6))
sns.catplot(x='education_level', hue='bank_account', kind='count', data=train)
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

plt.figure(figsize=(16, 6))
sns.catplot(x='job_type', hue='bank_account', kind='count', data=train)
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)

# function to preprocess our data from train models
def preprocessing_data(data):
    

    # Convert the following numerical labels from interger to float
    float_array = data[["household_size", "age_of_respondent",
                        "year"]].values.astype(float)

    # categorical features to be onverted to One Hot Encoding
    categ = ["relationship_with_head", "marital_status",
             "education_level", "job_type", "country"]

    # One Hot Encoding conversion
    data = pd.get_dummies(data, prefix_sep="_", columns=categ)

    # Label Encoder conversion
    data["location_type"] = le.fit_transform(data["location_type"])
    data["cellphone_access"] = le.fit_transform(data["cellphone_access"])
    data["gender_of_respondent"] = le.fit_transform(data["gender_of_respondent"])

    # drop uniquid column
    data = data.drop(["uniqueid"], axis=1)

    # scale our data into range of 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    return data

# function to preprocess our data from train models
def preprocessing_data(data):
    

    # Convert the following numerical labels from interger to float
    float_array = data[["household_size", "age_of_respondent",
                        "year"]].values.astype(float)

    # categorical features to be onverted to One Hot Encoding
    categ = ["relationship_with_head", "marital_status",
             "education_level", "job_type", "country"]

    # One Hot Encoding conversion
    data = pd.get_dummies(data, prefix_sep="_", columns=categ)

    # Label Encoder conversion
    data["location_type"] = le.fit_transform(data["location_type"])
    data["cellphone_access"] = le.fit_transform(data["cellphone_access"])
    data["gender_of_respondent"] = le.fit_transform(data["gender_of_respondent"])

    # drop uniquid column
    data = data.drop(["uniqueid"], axis=1)

    # scale our data into range of 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    return data

# view the first row of the processed_train dataset after preprocessing.
#Inclusive of Start, Exclusive of End
print(processed_train[:2])

# shape of the processed train set
print(processed_train.shape)

import sklearn.model_selection

# Split train_data
from sklearn.model_selection import train_test_split

X_Train, X_Val, y_Train, y_val = train_test_split(
    processed_train, y_train,
    stratify = y_train,
    test_size =28,
    random_state=42)

#import classifier algorithm here
from xgboost import XGBClassifier

# create models
xg_model = XGBClassifier()

#fitting the models
xg_model.fit(X_Train,y_Train)

#import classifier algorithm here
from xgboost import XGBClassifier

# create models
xg_model = XGBClassifier()

#fitting the models
xg_model.fit(X_Train,y_Train)


#print the classification report
from sklearn.metrics import classification_report

report = classification_report(y_val, xg_y_model)
print(report)


# calculate the accuracy and prediction of the model
from sklearn.metrics import accuracy_score, confusion_matrix,
ConfusionMatrixDisplay

xgboost_model_predicted = xg_model.predict(X_Val)
score = accuracy_score(y_val, xgboost_model_predicted)
print("Error rate for XGBClassifie model is: ", 1- score)
# Calculate confusion matrix
cm = confusion_matrix(y_val, xgboost_model_predicted,
                      normalize='true')
print("Confusion Matrix:")
print(cm)
# Plot confusion matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=np.unique(y_val))
disp.plot(cmap='viridis', values_format='.2f')
plt.title("Confusion Matrix")
plt.show()

# Get the predicted result for the test Data
test.bank_account = xg_model.predict(processed_test)

# Create submission DataFrame
submission = pd.DataFrame({
    "uniqueid": test["uniqueid"] + " x " + test["country"],
    "bank_account": test.bank_account
    })

#show the five sample
submission.sample(15)

#Create a Submission file in Jupyter notebook and download it for submission
from IPython.display import FileLink
submission.to_csv('submission1.csv', index=False)

#Dispaly a download link
FileLink('submission1.csv')

#in case of excel file
from IPython.display import FileLink
submission.to_excel('submission2.xlsx', index=False)

#Dispaly a download link
FileLink('submission2.xlsx')

