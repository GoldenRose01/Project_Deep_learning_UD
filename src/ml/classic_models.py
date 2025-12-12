from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class RandomForestModel:
    def __init__(self, n_trees=50):
        self.model = RandomForestClassifier(n_estimators=n_trees, max_depth=10)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)