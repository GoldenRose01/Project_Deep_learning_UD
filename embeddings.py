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
        self.vectorizer = None # Solo per TF-IDF
        
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
            
        elif self.method == 'tfidf