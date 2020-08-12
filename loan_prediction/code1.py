# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:34:29 2020

@author: amIT
"""
#important lib
import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
%matplotlib inline 
import warnings                        # To ignore any warnings 
warnings.filterwarnings("ignore")



#Reading data
train_data=pd.read_csv("train.csv") 
test_data=pd.read_csv("test.csv")

#copying orignal data
train_original=train_data.copy() 
test_original=test_data.copy()

#data understanding
train_data.columns
test_data.columns

#data type of every columns
train_data.dtypes

#shape
train_data.shape
test_data.shape

#univarite analysis
"""
In this section, we will do univariate analysis. It is the simplest form of analyzing data where we examine
each variable individually. For categorical features we can use frequency table or bar plots which will 
calculate the number of each category in a particular variable. For numerical features, probability 
density plots can be used to look at the distribution of the variable.
"""
#Target Variable
train_data['Loan_Status'].value_counts()
# Normalize can be set to True to print proportions instead of number
 train_data['Loan_Status'].value_counts(normalize=True)
 train_data['Loan_Status'].value_counts().plot.bar()
 
 #Independent Variable (Categorical)
plt.figure(1) 
plt.subplot(221)
train_data['Gender'].value_counts(normalize=True).plot.bar(figsize=(15,5), title= 'Gender') 
plt.subplot(222) 
train_data['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223) 
train_data['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224)
train_data['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()
 
#Independent Variable (Ordinal)
plt.figure(1)
plt.subplot(131)
train_data['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6),title='Dependents')
plt.subplot(132)
train_data['Education'].value_counts(normalize=True).plot.bar(figsize=(24,6),title= 'Education') 
plt.subplot(133) 
train_data['Property_Area'].value_counts(normalize=True).plot.bar(figsize=(24,6),title= 'Property_Area') 
plt.show()

#Independent Variable (Numerical)
plt.figure(1)
plt.subplot(121) 
sns.distplot(train_data['ApplicantIncome']); 
plt.subplot(122) 
train_data['ApplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()

train_data.boxplot(column='ApplicantIncome', by = 'Education') 
plt.suptitle("") 
Text(0.5,0.98,'')

plt.figure(1)
plt.subplot(121)
sns.distplot(train_data['CoapplicantIncome']);

plt.subplot(122)
train_data['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()

plt.figure(1)
plt.subplot(121)
df=train_data.dropna() 
sns.distplot(df['LoanAmount']); 
plt.subplot(122)
train_data['LoanAmount'].plot.box(figsize=(16,5)) 
plt.show()

#Bivariate Analysis
"""After looking at every variable individually in univariate analysis, we will now explore them again 
with respect to the target variable.
"""
#Categorical Independent Variable vs Target Variable
Gender=pd.crosstab(train_data['Gender'],train_data['Loan_Status']) 
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

#Now let us visualize the remaining categorical variables vs target variable.
Married=pd.crosstab(train_data['Married'],train_data['Loan_Status'])
Dependents=pd.crosstab(train_data['Dependents'],train_data['Loan_Status']) 
Education=pd.crosstab(train_data['Education'],train_data['Loan_Status']) 
Self_Employed=pd.crosstab(train_data['Self_Employed'],train_data['Loan_Status']) 

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show() 

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show()

Credit_History=pd.crosstab(train_data['Credit_History'],train_data['Loan_Status'])
Property_Area=pd.crosstab(train_data['Property_Area'],train_data['Loan_Status']) 

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show()

"""Numerical Independent Variable vs Target Variable
We will try to find the mean income of people for which the loan has been approved vs the mean income of people 
for which the loan has not been approved.
"""
train_data.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

"""
Here the y-axis represents the mean applicant income. We don’t see any change in the mean income.
 So, let’s make bins for the applicant income variable based on the values in it and analyze the 
 corresponding loan status for each bin.
"""
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train_data['Income_bin']=pd.cut(train_data['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(train_data['Income_bin'],train_data['Loan_Status']) 
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 
P = plt.ylabel('Percentage')

bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train_data['Coapplicant_Income_bin']=pd.cut(train_data['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train_data['Coapplicant_Income_bin'],train_data['Loan_Status']) 
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('CoapplicantIncome') 
P = plt.ylabel('Percentage')

train_data['Total_Income']=train_data['ApplicantIncome']+train_data['CoapplicantIncome']
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high']
train_data['Total_Income_bin']=pd.cut(train_data['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train_data['Total_Income_bin'],train_data['Loan_Status']) 
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Total_Income') 
P = plt.ylabel('Percentage')


#Let’s visualize the Loan amount variable.

bins=[0,100,200,700] 
group=['Low','Average','High'] 
train_data['LoanAmount_bin']=pd.cut(train_data['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train_data['LoanAmount_bin'],train_data['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')

train_data=train_data.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
train_data['Dependents'].replace('3+', 3,inplace=True) 
test_data['Dependents'].replace('3+', 3,inplace=True) 
train_data['Loan_Status'].replace('N', 0,inplace=True) 
train_data['Loan_Status'].replace('Y', 1,inplace=True)

#heat map to see relation btw numerical variable
matrix = train_data.corr() 
f, ax = plt.subplots(figsize=(9, 6)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
print(matrix)

#Missing value imputation
#Let’s list out feature-wise count of missing values.

train_data.isnull().sum()

#categorical 
train_data['Gender'].fillna(train_data['Gender'].mode()[0],inplace=True)
train_data['Married'].fillna(train_data['Married'].mode()[0], inplace=True) 
train_data['Dependents'].fillna(train_data['Dependents'].mode()[0], inplace=True) 
train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode()[0], inplace=True) 
train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0], inplace=True)

train_data['Loan_Amount_Term'].value_counts().plot.bar();
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mode()[0], inplace=True)
train_data['LoanAmount'].fillna(train_data['LoanAmount'].median(),inplace=True)
train_data.isnull().sum()

#filing missing value in test_data
test_data['Gender'].fillna(train_data['Gender'].mode()[0], inplace=True) 
test_data['Dependents'].fillna(train_data['Dependents'].mode()[0], inplace=True) 
test_data['Self_Employed'].fillna(train_data['Self_Employed'].mode()[0], inplace=True) 
test_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0], inplace=True) 
test_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mode()[0], inplace=True) 
test_data['LoanAmount'].fillna(train_data['LoanAmount'].median(), inplace=True)

#outliers ko sai karna by using log transformation
train_data['LoanAmount_log'] = np.log(train_data['LoanAmount']) 
train_data['LoanAmount_log'].hist(bins=20) 
test_data['LoanAmount_log'] = np.log(test_data['LoanAmount'])