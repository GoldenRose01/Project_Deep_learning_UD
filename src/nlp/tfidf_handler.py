from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfHandler:
    def __init__(self, max_features=5000):
        print("🧠 TF-IDF Initialized")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2) # Considera anche coppie di parole
        )
        self.matrix = None

    def encode(self, texts):
        print("🚀 TF-IDF Fitting...")
        self.matrix = self.vectorizer.fit_transform(texts)
        # Convertiamo in array denso solo se la RAM lo permette, altrimenti gestiamo sparso
        try:
            return self.matrix.toarray()
        except:
            print("⚠️ RAM insufficiente per array denso, restituisco matrice sparsa.")
            return self.matrix