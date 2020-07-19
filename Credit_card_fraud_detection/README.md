CREDIT CARD FRAUD DETECTION
 		
Name -AMIT BISHT
U. Roll No- 2012485
Sec-B (07)

PROBLEM STATEMENT: -  

The challenge is to recognize fraudulent credit card transactions so that the customers of credit card companies are not charged for items that they did not purchase.

ABSTRACT:-

Enormous Data is processed every day and the model build must be fast enough to respond to the scam in time.

Imbalanced Data i.e most of the transactions (99.8%) are not fraudulent which makes it really hard for detecting the fraudulent ones

Data contain 5rows and 31 column first column is time and other column are v1,v2…… they all contain numeric value.And in this project we detected a transaction as a fraud or not.

This can be done using random forest also but I used CNN to solve this real time problem.

Now why CNN because Multiple dimentional CNN are used for image data, 1D cnn can be using in NLP that is why I tried this approach.
Our objective will be to correctly classify the minority class of fraudulent transactions.

METHODOLOGY: -  

We will use Deep learning to solve this problem problem, we will preprocess the data with some standard technique to preprocces imbalanced data and after that we will train our data.

DATASET: -

The datasets contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

ALGORITHMS/TECHNIQUES:-

CNN is used.
CNNs are powerful image processing, artificial intelligence (AI) that use deep learning to perform both generative and descriptive tasks, often using machine vison that includes image and video recognition, along with recommender systems and natural language processing (NLP).

HOW AND WHAT I DID:-

Firstly I learned basics of Deep Learning from a course of Andrew Ng which I have purchased from Course Era .
Then I learn about this project from various Kaggle notebooks.

Preprocessing phase:-

Not much preproccesing was needed as the data was already clean but we divided the data set in two classes one fraud and other not fraud.

Made some new dataframe for the ease to acces data.

Extracted random entries of class-0 Total entries are 1.5* NO. of class-1 entries

Model fitting:-

Divided the data into Train-Test Split and Applied StandardScaler to obtain all the features in similar range after that we reshaped  the input to 3D used batch normalizations and created CNN model after that we Compiled and Fitied our model to the data set.

RESULT:-

We calculated results using epoch and accuracy was near 95%. 

Also plotted the train and validation graph for Accuracy and model loss.










