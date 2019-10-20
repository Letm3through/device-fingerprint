from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn import svm
from sklearn import metrics

# Read dataset to pandas dataframe
root_dir = os.path.abspath("./")
for file in os.listdir(root_dir):
    if "features.csv" in file:
        feature_file = file

full_feature_file_path = os.path.join(root_dir, feature_file)
gyro_data = pd.read_csv(full_feature_file_path)
labels = list(gyro_data['phone'])
gyro_data = gyro_data.drop("phone", axis=1)

X_train, X_test, y_train, y_test = train_test_split(gyro_data, labels, test_size=0.3)
clf = svm.SVC()

# Train the model using the training sets
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
# Model Accuracy: how often is the classifier correct
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# Evaluate model
test_features = os.path.abspath("./test.csv")
test_data = pd.read_csv(test_features)
labels = list(test_data['phone'])
test_data = test_data.drop("phone", axis=1)
result = clf.predict(test_data)
print(result)
print(confusion_matrix(labels, result))
print("Accuracy:", metrics.accuracy_score(labels, result))




