Predict Who Will Survive And Who Will Die
 		
Name -AMIT BISHT
U. Roll No-2012485
Sec-B (07)

PROBLEM STATEMENT: -  
  We will use the Titanic passenger data (name, age, price of ticket, etc) to try to predict who will survive and who will die.

ABSTRACT:-

We have a data set of  passenger on the ship titanic and we have to predict that who among them chould survive when ship will sink.
In this project I have made a machine learning model which will predict the survival of passenger on the basis of the attribute.
If the value of survived attribute in output file if-
•	if it's a "1", the passenger survived.
•	if it's a "0", the passenger died.
After the completion of project I uploaded the output file on Kaggle which results in 76.8% accuracy.

METHODOLOGY: -  

We will use machine learning techniques to solve a binary classification problem.
Random forest classifier is used to do so.


DATASET: -

train.csv contains the details of a subset of the passengers on board (891 passengers, to be exact -- where each passenger gets a different row in the table).
test.csv does not have a "Survived" column 
(we need to predict this)

ALGORITHMS/TECHNIQUES:-

In this project, Random Forrest (RF) will be implemented and the learning algorithms will be implemented by sklearn toolbox in this project.

RANDOM FORREST:-
RF is an ensemble classifier, it fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. sklearn provides a function called RandomForestClassifier(). One typical tunable parameter is called ”n estimators”, which represents the number of trees in the forest.

HOW AND WHAT I DID:-

Firstly I learned basics of machine learning from a course which I have purchased from udemy.
Then I learn machine learning from Kaggle short courses.
This project was part of that course first I finished that and than I start this project to get more accuracy by pre-processing data and applying the suitable ML algorithm.

Preprocessing phase:-

Basically in pre-processing phase I changed column Embarked to numerical format as pre-processing in string is difficult and whit numerical format of data it will be easy to get co-relations btw attributes.
And changed sex column to 0,1 male=0, female=1.

EDA:-

Some sort of EDA is done on the basis of survived people , their age, their on board location and ages are classified as class eg: class1, class 2, class 3.

Model fitting:-

Random forest model was fitted on features_forest(which is basically our training set which contain 8 attributes. And our training set was target which contain survived attribute.
Than we printed our model score by invoking score method. We also used model_selection library of sklearn to calculate accuracy .

RESULT:-

Than finally we fitted our model on test set and predicted the result and stored it in a file named as random_forest.csv and submitted on kaggle. Our accuracy was 76.8.
Form the result I concluded that male chances were more to survive than woman and young people who were in class 1 were having surviving chances high.









