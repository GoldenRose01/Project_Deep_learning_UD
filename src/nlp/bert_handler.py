import torch
from sentence_transformers import SentenceTransformer

class BertHandler:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🧠 BERT Device: {self.device.upper()}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts):
        return self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True
        )