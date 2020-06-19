from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = 0.1, random_state = 123)

forest = RandomForestClassifier(n_estimators=8)
forest = forest.fit(train_X, train_Y)
print(forest.score(test_X, test_Y))

# Output: 1.0

rf_output = forest.predict(test_X)
print(rf_output)

# Output: [1 2 2 1 0 2 1 0 0 1 2 0 1 2 2]
