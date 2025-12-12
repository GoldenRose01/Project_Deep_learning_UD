import numpy as np
import torch
import os
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingGenerator:
    def __init__(self, method='bert', model_name='all-MiniLM-L6-v2'):
        self.method = method
        self.model = None
        self.vectorizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- FIX: CREAZIONE CARTELLA ---
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_path = os.path.join(self.cache_dir, f"embeddings_{method}.npy")

        print(f"🔄 Init AI ({method.upper()} on {self.device})")

        if self.method == 'bert':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=self.device)
        elif self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    def fit_transform(self, text_list):
        # 1. CONTROLLO CACHE
        if os.path.exists(self.cache_path):
            try:
                data = np.load(self.cache_path)
                # Verifica rudimentale della lunghezza
                if len(data) == len(text_list):
                    print(f"⚡ EMBEDDINGS TROVATI: Carico da {self.cache_path}")
                    return data
                else:
                    print(f"⚠️ Cache mismatch ({len(data)} vs {len(text_list)}). Ricalcolo...")
            except:
                pass

        if not text_list: return None
        print(f"🚀 GENERAZIONE NUOVI VETTORI ({len(text_list)} items)...")

        embeddings = None
        if self.method == 'bert':
            embeddings = self.model.encode(
                text_list,
                show_progress_bar=True,
                batch_size=64,
                convert_to_numpy=True
            )

        elif self.method == 'tfidf':
            embeddings = self.vectorizer.fit_transform(text_list).toarray()

        # 2. SALVATAGGIO
        if embeddings is not None:
            print(f"💾 Salvataggio Embeddings in {self.cache_path}...")
            np.save(self.cache_path, embeddings)

        return embeddings