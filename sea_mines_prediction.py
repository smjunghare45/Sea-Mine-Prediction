
'''This is an ML project for prediction of mines in the sea. It helps to predict whether the article found is metal(mine) or a rock. This is predicted using logistic regression model.'''
#Importing the dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# data collection and preprocessing

sonar_df = pd.read_csv('/content/sonar data.csv', header = None)    #since the dataset has no column heading "header = None"

sonar_df.head()

# number of rows and cols (Rows, cols)

sonar_df.shape

# how many R and how many M (i.e Checking for imbalance dataset)

# if the cols do not have heading in the dataset or dataframes create call it by using it's index....

sonar_df[60].value_counts()

sonar_df.describe()       # (Statistical Measures) This gives count, mean, std deviation, etc of every column..

sonar_df.groupby(60).mean()

#seperating data and labels (This is done for supervised learning models)

X = sonar_df.drop(columns=60, axis=1)
Y = sonar_df[60]

print(X)
print(Y)

# TRAIN - TEST SPLIT

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# test_size = 0.1 means keeping 10% of data for testing rest of it for training
# stratify = Y means our data will be splitted on based on Y i.e R and M
# random_state = 1 means to split data in a particular order

print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

# MODEL TRAINING ----> using a Logistic Regression Model

model = LogisticRegression()

# training  the model with training data

model.fit(X_train, Y_train)

# Model Evaluation -- accuracy on training data

X_train_predict = model.predict(X_train)
train_data_accu = accuracy_score(X_train_predict, Y_train)    # comparison between prediction of our model and the original label of the data

print('Accuracy on training data :', train_data_accu)

"""Hence, we got the accuracy of 83.42% for our training data"""

# Model Evaluation -- accuracy on test data

X_test_predict = model.predict(X_test)
test_data_accu = accuracy_score(X_test_predict, Y_test)  # comparison between prediction of our model and the original label of the test data

print('Accuracy on test data :', test_data_accu)

"""Hence, we got the accuracy of 76% for our test data"""

### TILL NOW WE HAVE TRAINED THE MODEL BUT NOW WE NEED TO PREDICT THAT WHETHER THE ARTICLE IS ROCK OR MINE###

# PREDICTIVE SYSTEM #

input_data = (0.0187,0.0346,0.0168,0.0177,0.0393,0.1630,0.2028,0.1694,0.2328,0.2684,0.3108,0.2933,0.2275,0.0994,0.1801,0.2200,0.2732,0.2862,0.2034,0.1740,0.4130,0.6879,0.8120,0.8453,0.8919,0.9300,0.9987,1.0000,0.8104,0.6199,0.6041,0.5547,0.4160,0.1472,0.0849,0.0608,0.0969,0.1411,0.1676,0.1200,0.1201,0.1036,0.1977,0.1339,0.0902,0.1085,0.1521,0.1363,0.0858,0.0290,0.0203,0.0116,0.0098,0.0199,0.0033,0.0101,0.0065,0.0115,0.0193,0.0157)

#changing the data type list to numpy array --> this is done beacuse processing on numpy array is faster and easier..
input_data_to_numpy = np.asarray(input_data)

# reshaping the numpy array --> because we are predicting it for one instance
input_data_reshape = input_data_to_numpy.reshape(1,-1)

prediction = model.predict(input_data_reshape)
print(prediction)

if (prediction[0]=='M'):
  print('The article found is a Mine')
else:
  print('The article found is a Rock')

