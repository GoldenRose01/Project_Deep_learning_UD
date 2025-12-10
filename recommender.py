import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, df, embeddings):
        """
        Inizializza il sistema di raccomandazione.
        :param df: DataFrame pandas con i dati dei film
        :param embeddings: Matrice numpy degli embedding pre-calcolati
        """
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.sim_matrix = None
        
    def compute_similarity(self):
        """Calcola la matrice di similarità del coseno (N x N)."""
        print("🧮 Calcolo della Similarità del Coseno...")
        self.sim_matrix = cosine_similarity(self.embeddings)
        print("✅ Matrice di similarità calcolata.")

    def recommend(self, movie_title, top_n=3):
        """
        Raccomanda film simili a quello dato in input.
        """
        # 1. Trova l'indice del film (Case insensitive)
        movie_title_lower = movie_title.lower()
        matches = self.df[self.df['title'].str.lower() == movie_title_lower]
        
        if matches.empty:
            return None, f"Film '{movie_title}' non trovato nel database."
        
        idx = matches.index[0]
        
        # 2. Ottieni i punteggi di similarità per questo film
        scores = list(enumerate(self.sim_matrix[idx]))
        
        # 3. Ordina (dal più alto al più basso)
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # 4. Prendi i primi N (saltando il primo che è il film stesso)
        top_scores = scores[1:top_n+1]
        
        # 5. Costruisci il risultato
        results = []
        for i, score in top_scores:
            row = self.df.iloc[i]
            results.append({
                'title': row['title'],
                'genres': row['genres'],
                'score': round(score, 4),
                'overview': row['overview'][:100] + "..." # Tronca la trama per leggibilità
            })
            
        return pd.DataFrame(results), "OK"