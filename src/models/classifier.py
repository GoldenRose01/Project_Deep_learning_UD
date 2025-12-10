from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


class GenreClassifier:
    def __init__(self, embeddings, labels):
        """
        embeddings: Matrice X (Features da BERT)
        labels: Lista y (Target: Genere, Tipo o Sorgente)
        """
        self.X = embeddings
        self.y = labels
        self.model = None
        # Dividiamo i dati: 80% per addestrare, 20% per il test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train(self, epochs=10, hidden_layers=(64, 32)):
        """
        Addestra una rete neurale (MLP).
        epochs: corrisponde a max_iter in sklearn (numero di passaggi sui dati)
        """
        # Se l'utente mette poche epoche, usiamo quelle. Se ne mette tante, lasciamo lavorare.
        print(f"🧠 Training Neural Network per {epochs} epoche...")

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=epochs,  # Qui usiamo il numero scelto dallo slider!
            activation='relu',
            solver='adam',
            random_state=42,
            early_stopping=True  # Si ferma prima se smette di imparare
        )

        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        """Restituisce l'accuratezza e il report dettagliato."""
        if not self.model:
            return 0, "Model not trained"

        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        # Output dict=True serve per trasformarlo in tabella su Streamlit
        return acc, classification_report(self.y_test, preds, output_dict=True, zero_division=0)