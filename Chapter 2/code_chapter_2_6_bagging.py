from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

#divide data into train and test set
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state = 123)

clf = BaggingClassifier(base_estimator=SVC(),
                        n_estimators=10, random_state=0).fit(X_train, y_train)

print(clf.score(X_test, y_test))

# Output: 0.9