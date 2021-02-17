#Importing dataset
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# INITIALIZING CLASSIFIER
clf = RandomForestClassifier()
best = 0

# TRAINING MULTIPLE CLASSIFIERS AND PICKING THE BEST
for i in range(10):

    clf.fit(x_train, y_train)

    # GET MODEL ACCURACY
    acc = clf.score(x_test, y_test) * 100

    # IF CURRENT MODEL HAS A BETTER SCORE SAVE IN A PICKLE FILE
    if acc > best:
        best = acc
        with open("model.pickle", "wb") as f:
            pickle.dump(clf, f)
            print(acc)


# LOAD MODEL
pickle_in = open("model.pickle", "rb")
model = pickle.load(pickle_in)

acc2 = model.score(x_test, y_test) * 100
print(acc2)
