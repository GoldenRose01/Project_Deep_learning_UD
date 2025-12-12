import numpy as np
import torch
import os
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingGenerator:
    def __init__(self, method='bert', model_name='all-MiniLM-L6-v2'):
        self.method = method
        self.model = None
        self.vectorizer = None

        # LOGICA GPU AVANZATA
        if torch.cuda.is_available():
            self.device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU RILEVATA: {gpu_name} (Modalità Turbo Attiva)")
        else:
            self.device = 'cpu'
            print("⚠️ GPU NON RILEVATA: Uso CPU (Più lento)")

        # Creazione cartella Cache
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"embeddings_{method}.npy")

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
                if len(data) == len(text_list):
                    print(f"⚡ EMBEDDINGS CACHED: {self.cache_path}")
                    return data
            except:
                pass

        if not text_list: return None
        print(f"🔥 INIZIO CALCOLO SU {self.device.upper()} ({len(text_list)} film)...")

        embeddings = None
        if self.method == 'bert':
            # Batch size più alto per la tua RTX (sfrutta la VRAM)
            embeddings = self.model.encode(
                text_list,
                show_progress_bar=True,
                batch_size=128,  # Aumentato per la tua GPU
                convert_to_numpy=True
            )

        elif self.method == 'tfidf':
            embeddings = self.vectorizer.fit_transform(text_list).toarray()

        # 2. SALVATAGGIO
        if embeddings is not None:
            np.save(self.cache_path, embeddings)

        return embeddings