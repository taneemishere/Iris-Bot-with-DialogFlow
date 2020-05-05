from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the data
iris = datasets.load_iris()

X = iris.data
y = iris.target


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define and fit to the model

clf = RandomForestClassifier(n_estimators=10)

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

# print(predicted)
print(accuracy_score(predicted, y_test))
print(clf.predict(X_test))


# Saving the model as Pickle

with open(r'rf.pkl','wb') as model_pkl:
    pickle.dump(clf, model_pkl, protocol=2)

# open a file, where you want to store the data
# model_pkl = open('rf.pkl', 'wb')
# dump information to that file
# pickle.dump(clf, model_pkl, protocol=2)
# close the file
# model_pkl.close()


# Our pickled model 'rf.pkl' will be loaded in flask app