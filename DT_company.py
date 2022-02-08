"""Decision Tree

Assignment


About the data:
Let’s consider a Company dataset with around 10 variables and 400 records.
The attributes are as follows:
 Sales -- Unit sales (in thousands) at each location
 Competitor Price -- Price charged by competitor at each location
 Income -- Community income level (in thousands of dollars)
 Advertising -- Local advertising budget for company at each location (in thousands of dollars)
 Population -- Population size in region (in thousands)
 Price -- Price company charges for car seats at each site
 Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
 Age -- Average age of the local population
 Education -- Education level at each location
 Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
 US -- A factor with levels No and Yes to indicate whether the store is in the US or not
The company dataset looks like this:

Problem Statement:
A cloth manufacturing company is interested to know about the segment or attributes causes high sale.
Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) & all other variable will be independent in the analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

company = pd.read_csv('C:/ExcelrData/Data-Science_Assignments/Decision_Trees/Company_Data.csv')
company.columns
company.Sales.median()
company.isna().sum()

# create bins for sales
cut_labels = ['Low', 'Medium', 'High']
cut_bins = [-1, 5.66, 12, 17]
company['sales'] = pd.cut(company['Sales'], labels=cut_labels, bins=cut_bins)

company.pop('Sales')

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
company['ShelveLoc'] = label_encoder.fit_transform(company['ShelveLoc'])
company['Urban'] = label_encoder.fit_transform(company['Urban'])
company['US'] = label_encoder.fit_transform(company['US'])

col_names = list(company.columns)
predictors = col_names[0:10]
target = col_names[10]

from sklearn.model_selection import train_test_split

train, test = train_test_split(company, test_size=0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier as DS

model = DS(criterion='entropy')
model.fit(train[predictors], train[target])
train_pred = model.predict(train[predictors])
test_pred = model.predict(test[predictors])

train_acc = np.mean(train_pred == train[target])
test_acc = np.mean(test_pred == test[target])
print(train_acc)
print(test_acc)
