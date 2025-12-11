import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer


# Per installare le librerie necessarie, eseguite nel terminale:
# pip install scikit-learn sentence-transformers numpy pandas

class EmbeddingGenerator:
    """
    Classe per generare rappresentazioni vettoriali (embeddings) delle trame dei film.
    Supporta due modalità:
    1. 'tfidf': Baseline classica (Frequency-based).
    2. 'bert': Approccio avanzato (Context-based) usando Sentence-BERT.
    """

    def __init__(self, method='bert', model_name='all-MiniLM-L6-v2'):
        """
        Inizializza il generatore.
        Args:
            method (str): 'tfidf' oppure 'bert'.
            model_name (str): Nome del modello HuggingFace (solo per BERT).
                              'all-MiniLM-L6-v2' è veloce e ottimo per la similarità.
        """
        self.method = method
        self.model = None
        self.vectorizer = None  # Solo per TF-IDF

        print(f"🔄 Inizializzazione EmbeddingGenerator con metodo: {self.method.upper()}...")

        if self.method == 'bert':
            try:
                from sentence_transformers import SentenceTransformer
                # Scarica il modello automaticamente la prima volta
                self.model = SentenceTransformer(model_name)
            except ImportError:
                raise ImportError("Per usare BERT devi installare la libreria: pip install sentence-transformers")

        elif self.method == 'tfidf':
            # Rimuoviamo stop words inglesi e limitiamo a 5000 parole chiave per efficienza
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

        else:
            raise ValueError("Metodo non supportato. Scegli tra 'tfidf' o 'bert'.")

        print("✅ Inizializzazione completata.")

    def fit_transform(self, text_list):
        """
        Genera gli embedding per una lista di testi.
        Args:
            text_list (list): Lista di stringhe (le trame dei film).
        Returns:
            numpy.ndarray: Matrice degli embeddings.
        """
        if not text_list:
            raise ValueError("La lista delle trame è vuota!")

        print(f"🚀 Generazione embeddings in corso per {len(text_list)} trame...")

        if self.method == 'bert':
            # Sentence-BERT gestisce tutto internamente (progress bar inclusa)
            embeddings = self.model.encode(text_list, show_progress_bar=True)

        elif self.method == 'tfidf':
            # TF-IDF deve prima imparare il vocabolario (fit) poi trasformare
            embeddings = self.vectorizer.fit_transform(text_list).toarray()

        print(f"✅ Embeddings generati! Dimensione matrice: {embeddings.shape}")
        return embeddings

    def save_embeddings(self, embeddings, filepath):
        """Salva gli embedding su disco per non ricalcolarli ogni volta."""
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"💾 Embeddings salvati in: {filepath}")

    def load_embeddings(self, filepath):
        """Carica gli embedding dal disco."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File non trovato: {filepath}")

        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"📂 Embeddings caricati da: {filepath}")
        return embeddings


# ==========================================
# AREA DI TEST (Esegui questo file per provare)
# ==========================================
if __name__ == "__main__":
    # Dati finti per testare subito se VS Code e le librerie funzionano
    trame_test = [
        "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
        # Toy Story
        "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        # Matrix
        "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        # The Godfather
        "A space toy action figure lands in a child's room and believes he is a real space ranger."
        # Trama inventata simile a Toy Story
    ]

    # PROVA 1: TF-IDF (Baseline)
    print("\n--- TEST BASELINE (TF-IDF) ---")
    generator_tfidf = EmbeddingGenerator(method='tfidf')
    vecs_tfidf = generator_tfidf.fit_transform(trame_test)
    print(f"Esempio vettore (primi 5 valori): {vecs_tfidf[0][:5]}")

    # PROVA 2: BERT (Avanzato - Quello che vogliono i prof)
    print("\n--- TEST AVANZATO (BERT/SBERT) ---")
    generator_bert = EmbeddingGenerator(method='bert')
    vecs_bert = generator_bert.fit_transform(trame_test)
    print(f"Esempio vettore (primi 5 valori): {vecs_bert[0][:5]}")

    # Se vedi numeri stampati nel terminale, tutto funziona!