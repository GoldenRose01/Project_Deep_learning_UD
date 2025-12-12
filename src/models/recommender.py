import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, df, embeddings):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        # Mappa titolo -> indice per velocità
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()

    def get_profile_recommendations(self, movie_titles, filter_type="All", top_n=5):
        """
        Genera raccomandazioni basate su una LISTA di film (Profilo Utente).
        """
        valid_indices = []
        for title in movie_titles:
            if title in self.indices:
                idx = self.indices[title]
                # Se ci sono duplicati, prende il primo
                if isinstance(idx, pd.Series): idx = idx.iloc[0]
                valid_indices.append(idx)

        if not valid_indices:
            return None, "Nessun film valido selezionato."

        # 1. Calcola il "Vettore Utente" (Media degli embedding dei film scelti)
        selected_embeddings = self.embeddings[valid_indices]
        user_profile_vector = np.mean(selected_embeddings, axis=0).reshape(1, -1)

        # 2. Calcola similarità tra il profilo utente e TUTTI i film
        # Nota: usiamo cosine_similarity dinamico qui perché il vettore utente cambia sempre
        sim_scores = cosine_similarity(user_profile_vector, self.embeddings)[0]

        # 3. Crea un DataFrame temporaneo per filtrare
        temp_df = self.df.copy()
        temp_df['score'] = sim_scores

        # 4. Filtra per TIPO (Movie vs TV Show)
        if filter_type == "Movies Only":
            temp_df = temp_df[temp_df['type'] == 'Movie']
        elif filter_type == "TV Shows Only":
            temp_df = temp_df[temp_df['type'] == 'TV Show']

        # 5. Rimuovi i film che l'utente ha già inserito nel profilo (non consigliargli ciò che ha già visto)
        temp_df = temp_df[~temp_df['title'].isin(movie_titles)]

        # 6. Ordina e prendi i Top N
        results = temp_df.sort_values(by='score', ascending=False).head(top_n)

        return results, "OK"

    def evaluate_system_quality(self, n_samples=50):
        """
        Esegue un test automatico su 50 film casuali.
        Controlla se i film raccomandati condividono i generi con il film di input.
        """
        # Prendi 50 film a caso
        samples = self.df.sample(n=min(n_samples, len(self.df)), random_state=42)

        total_overlap_score = 0
        total_recs = 0

        for _, row in samples.iterrows():
            input_genres = set(str(row['genres']).split('|'))

            # Chiedi raccomandazioni
            recs, _ = self.recommend(row['title'], top_n=5)

            if recs is None or recs.empty: continue

            # Calcola overlap
            for _, rec_row in recs.iterrows():
                rec_genres = set(str(rec_row['genres']).split('|'))
                # Intersezione: quanti generi in comune?
                common = input_genres.intersection(rec_genres)
                if len(common) > 0:
                    total_overlap_score += 1  # Un punto se c'è almeno un genere in comune
                total_recs += 1

        # Calcola percentuale finale
        precision = total_overlap_score / total_recs if total_recs > 0 else 0
        return precision