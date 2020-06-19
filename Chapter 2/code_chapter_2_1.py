from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)

train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = 0.2, random_state = 123)

tree = DecisionTreeClassifier()    
tree.fit(train_X, train_Y)
print(tree.score(test_X, test_Y))

# Output: 0.9333333333333333