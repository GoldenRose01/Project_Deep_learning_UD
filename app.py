import streamlit as st
from src.data.ingestion import DataIngestor
from src.nlp.bert_handler import BertHandler
from src.models.embeddings import EmbeddingGenerator  # Usiamo la classe aggiornata sopra
from src.algorithms.content_based import ContentBasedRecommender
from src.services.web_search import WebSearchService
from src.services.translation import TranslationService
from src.ui.layout import render_main_page

st.set_page_config(layout="wide", page_title="Movie AI Pro")


@st.cache_resource
def init_backend():
    # 1. Dati (Carica da Cache se esiste)
    df = DataIngestor().load_all()

    # 2. NLP (Carica da Cache se esiste)
    texts = (df['title'].astype(str) + ". " + df['overview'].astype(str)).tolist()

    # Qui usiamo la classe EmbeddingGenerator che abbiamo appena modificato
    # (Sostituisce BertHandler diretto per gestire il caching)
    embedder = EmbeddingGenerator(method='bert')
    embeddings = embedder.fit_transform(texts)

    # 3. Core
    recsys = ContentBasedRecommender(df, embeddings)
    web = WebSearchService()
    trans = TranslationService()

    return df, recsys, web, trans, embeddings


with st.spinner("Avvio Sistema (Controllo Cache)..."):
    df, recsys, web, trans, embeddings = init_backend()

render_main_page(df, recsys, web, trans, embeddings)