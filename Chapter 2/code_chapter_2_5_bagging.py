from sklearn.utils import resample
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np 
from sklearn.metrics import accuracy_score

# data to be sampled
n_samples = 100
X,y = make_classification(n_samples=n_samples, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

#X, y = make_classification(n_samples=3 * n_samples, n_features=20, random_state=42)


#divide data into train and test set
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.1, random_state = 123)

# Number of divisions needed
num_divisions = 3
list_of_data_divisions = []
# Divide data into divisions 
for x in range(0, num_divisions):
    X_train_sample, y_train_sample = resample(X_train, y_train, replace=True, n_samples=7)
    sample = [X_train_sample, y_train_sample]
    list_of_data_divisions.append(sample)

#print(list_of_data_divisions)
# Learn a Classifier for each data divisions
learners = []
for data_division in list_of_data_divisions:
    data_x = data_division[0]
    data_y = data_division[1]
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(data_x, data_y)
    learners.append(decision_tree)

# Combine output of all classifiers using voting 
predictions = []
for i in range(len(y_test)):
    counts = [0 for _ in range(num_divisions)]
    for j , learner in enumerate(learners):
        prediction = learner.predict([X_test[i]])
        if prediction == 1:
            counts[j] = counts[j] + 1
    final_predictions = np.argmax(counts)
    predictions.append(final_predictions)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
# Output: Accuracy: 0.9
