from sklearn.neural_network import MLPClassifier


class NeuralNetModel:
    def __init__(self, hidden_layers=(64, 32), epochs=50):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=epochs,
            activation='relu',
            solver='adam',
            random_state=42
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)