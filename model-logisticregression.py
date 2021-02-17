#Importing dataset
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle

# GETTING COLUMNS FROM DATA SET
data = pd.read_csv('train.csv')[['Survived','Pclass','Age','Sex']]

# REMOVING NaN VAALUES
data = data.dropna()

# SPLITING DATA SET
X = data.drop(['Survived'], 1)
Y = data['Survived']

# ENCODING SEX COLUMN
le = preprocessing.LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])

# CREATING TRAIN AND TEST SAMPLES
x_train, x_test, y_train, y_test =  train_test_split(X, Y, test_size = 0.1)

best = 0

for i in range(10):

    model = LogisticRegression()

    model.fit(x_train, y_train)

    # GET MODEL ACCURACY
    acc = model.score(x_test, y_test) * 100
    print(acc)

    # IF CURRENT MODEL HAS A BETTER SCORE SAVE IN A PICKLE FILE
    if acc > best:
        best = acc
        with open("model-log.pickle", "wb") as f:
            pickle.dump(model, f)
            print('saved')


# LOAD MODEL
pickle_in = open("model-log.pickle", "rb")
model = pickle.load(pickle_in)


# GET MODEL ACCURACY
acc = model.score(x_test, y_test) * 100

print(acc)


#USED IN KAGGLE SUBMISSION

# GET TEST DATA
test = pd.read_csv('test.csv')[['Pclass','Age','Sex']]

# ENCODING SEX COLUMN
le = preprocessing.LabelEncoder()
test['Sex'] = le.fit_transform(test['Sex'])

import numpy as np

#REPLACE NaN VALUES WITH DEFAULT 0.0
test = test.replace( np.nan, 0)

# GETTING PREDICTIONS
predictions = model.predict(test.values)

pred = pd.concat([pd.read_csv('test.csv')[['PassengerId']], pd.DataFrame(predictions, columns=['Survived'])], axis = 1)

# pred.to_csv('submission-log.csv',  sep=',', index=False)