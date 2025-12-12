import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .neural_nets import NeuralNetModel
from .classic_models import RandomForestModel


class BenchmarkRunner:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def run(self):
        models = [
            ("NN (Simple)", NeuralNetModel(hidden_layers=(32,))),
            ("NN (Deep)", NeuralNetModel(hidden_layers=(128, 64))),
            ("Random Forest", RandomForestModel(n_trees=50))
        ]

        results = []
        for name, model_cls in models:
            start = time.time()
            model_cls.train(self.X_train, self.y_train)
            duration = time.time() - start

            acc = accuracy_score(self.y_test, model_cls.predict(self.X_test))
            results.append({"Model": name, "Accuracy": acc, "Time (s)": duration})

        return results