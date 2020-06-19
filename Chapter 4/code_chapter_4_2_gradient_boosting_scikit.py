from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

X, y = make_hastie_10_2(random_state=0)
clf = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
).fit(X, y)

scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())
# Output: 0.9225
