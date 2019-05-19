## Locate the working directory
import os
path = 'F:\zAssignments\SME Classification'
os.chdir(path)

## Import package for Data Reading
import pandas as pd

# Import the dataset
dataset = pd.read_excel('SME Data.xlsx')
# Encoding the nominal columns
df0 = pd.get_dummies(dataset[['State', 'Gender', 'Profession', 'Subj.Domain']])
df = pd.concat([dataset[['Name']], df0, dataset[['Age', 'Work.Exper', 'Event.Level']]], axis = 1)
X = df.iloc[:, 1:len(df.columns)].values
y = (dataset.iloc[:, 8].values > 0.5)*1

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

## Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver = 'lbfgs')
classifier.fit(X_train, y_train)

## Predicting the Test set results
y_pred = classifier.predict(X_test)

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
