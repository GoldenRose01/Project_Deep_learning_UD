import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, df, embeddings):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        # Mappa veloce lower_title -> index
        self.indices = pd.Series(self.df.index, index=self.df['title'].str.lower()).drop_duplicates()

    def recommend_single(self, title, top_n=5):
        idx = self.indices.get(title.lower())
        if idx is None: return None
        if isinstance(idx, pd.Series): idx = idx.iloc[0]

        # reshape(1, -1) per singolo vettore
        target = self.embeddings[idx].reshape(1, -1)

        # Similarità vettoriale (Veloce)
        scores = cosine_similarity(target, self.embeddings).flatten()

        top_idx = scores.argsort()[-(top_n + 1):-1][::-1]

        res = []
        for i in top_idx:
            row = self.df.iloc[i].to_dict()
            row['score'] = scores[i]
            res.append(row)
        return pd.DataFrame(res)

    def recommend_profile(self, titles, top_n=5):
        valid_idxs = [self.indices.get(t.lower()) for t in titles if self.indices.get(t.lower()) is not None]
        # Clean series results
        valid_idxs = [i.iloc[0] if isinstance(i, pd.Series) else i for i in valid_idxs]

        if not valid_idxs: return None

        # Media vettoriale
        user_vec = np.mean(self.embeddings[valid_idxs], axis=0).reshape(1, -1)
        scores = cosine_similarity(user_vec, self.embeddings).flatten()

        top_idx = scores.argsort()[::-1]

        res = []
        input_lower = [t.lower() for t in titles]

        count = 0
        for i in top_idx:
            if self.df.iloc[i]['title'].lower() in input_lower: continue
            row = self.df.iloc[i].to_dict()
            row['score'] = scores[i]
            res.append(row)
            count += 1
            if count >= top_n: break

        return pd.DataFrame(res)