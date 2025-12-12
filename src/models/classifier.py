import time
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class ModelBenchmark:
    def __init__(self, embeddings, labels):
        self.X = embeddings
        self.y = labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def run_benchmark(self):
        """Confronta diverse architetture"""
        models = [
            {
                "name": "Neural Net (Simple)",
                "model": MLPClassifier(hidden_layer_sizes=(32,), max_iter=50)
            },
            {
                "name": "Neural Net (Deep)",
                "model": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=50)
            },
            {
                "name": "Decision Tree (Fast)",
                "model": DecisionTreeClassifier(max_depth=10)
            },
            {
                "name": "Random Forest (Robust)",
                "model": RandomForestClassifier(n_estimators=50, max_depth=10)
            }
        ]

        results = []
        for entry in models:
            name = entry["name"]
            clf = entry["model"]

            print(f"Testing {name}...")

            # Misura Tempo Training
            start_time = time.time()
            clf.fit(self.X_train, self.y_train)
            train_time = time.time() - start_time

            # Misura Accuratezza
            preds = clf.predict(self.X_test)
            acc = accuracy_score(self.y_test, preds)

            results.append({
                "Modello": name,
                "Accuratezza": round(acc, 4),
                "Tempo Training (sec)": round(train_time, 4),
                "Efficienza (Acc/Time)": round(acc / (train_time + 0.001), 2)
            })

        return results