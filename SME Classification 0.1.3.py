## Locate the working directory
import os
path = 'F:\zAssignments\SME Classification'
os.chdir(path)

## Import Packages pandas and numpy
import pandas as pd
import numpy as np
# Read the dataset
dataset = pd.read_excel('SME Data.xlsx')

# Encoding the nominal columns
df0 = pd.get_dummies(dataset[['State', 'Gender', 'Profession', 'Subj.Domain']])
df = pd.concat([dataset[['Name']], df0, dataset[['Age', 'Work.Exper', 'Event.Level']]], axis = 1)
X = df.iloc[:, 1:len(df.columns)].values
# Transform the dependent variable 'Event.Rel' into binary type
y = (dataset.iloc[:, dataset.columns.get_loc("Event.Rel")].values > 0.5)*1

# Encoding the ordinal column Event.Level
from sklearn.preprocessing import LabelEncoder
labelencoder_X_ = LabelEncoder()
X[:, (len(df.columns)-2)] = labelencoder_X_.fit_transform(X[:, (len(df.columns)-2)])
X = X.astype(float)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Develop ANN model
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialization of ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = np.round((len(df.columns)-1)/2).astype(int), kernel_initializer = 'uniform', activation = 'relu', input_dim = (len(df.columns)-1)))
# Adding the second hidden layer
classifier.add(Dense(units = np.round((len(df.columns)-1)/2).astype(int), kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

## Compile the ANN model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Fit the ANN model to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 25)

## Prediction of Test Set result
y_pred = (classifier.predict(X_test) > 0.5)*1
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
