import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, df, embeddings):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        # Creiamo un indice tutto minuscolo per la ricerca esatta veloce
        self.indices = pd.Series(self.df.index, index=self.df['title'].str.lower()).drop_duplicates()

    def recommend_single(self, title, top_n=5):
        # Cerca nel dizionario lower-case
        idx = self.indices.get(title.lower())

        if idx is None: return None
        if isinstance(idx, pd.Series): idx = idx.iloc[0]

        target = self.embeddings[idx].reshape(1, -1)
        scores = cosine_similarity(target, self.embeddings).flatten()

        # Ordina
        top_idx = scores.argsort()[-(top_n + 1):-1][::-1]

        res = []
        for i in top_idx:
            row = self.df.iloc[i].to_dict()
            row['score'] = scores[i]
            res.append(row)
        return pd.DataFrame(res)

    def recommend_profile(self, titles, top_n=5):
        # Ottieni gli indici validi
        valid_idxs = []
        for t in titles:
            ix = self.indices.get(t.lower())
            if ix is not None:
                if isinstance(ix, pd.Series): ix = ix.iloc[0]
                valid_idxs.append(ix)

        if not valid_idxs: return None

        # Calcola Media Vettoriale
        user_vec = np.mean(self.embeddings[valid_idxs], axis=0).reshape(1, -1)
        scores = cosine_similarity(user_vec, self.embeddings).flatten()

        top_idx = scores.argsort()[::-1]

        res = []
        # Lista di titoli input in minuscolo per esclusione
        input_lower = [t.lower() for t in titles]

        count = 0
        for i in top_idx:
            curr_title = self.df.iloc[i]['title']
            # Non raccomandare ciò che l'utente ha già selezionato
            if curr_title.lower() in input_lower: continue

            row = self.df.iloc[i].to_dict()
            row['score'] = scores[i]
            res.append(row)
            count += 1
            if count >= top_n: break

        return pd.DataFrame(res)