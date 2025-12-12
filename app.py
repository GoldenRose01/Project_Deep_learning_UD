import streamlit as st
from src.data.ingestion import DataIngestor
from src.nlp.bert_handler import BertHandler
from src.algorithms.content_based import ContentBasedRecommender
from src.services.web_search import WebSearchService
from src.services.translation import TranslationService
from src.ui.layout import render_main_page

st.set_page_config(layout="wide", page_title="Movie AI Pro")


@st.cache_resource
def init_backend():
    # 1. Dati
    df = DataIngestor().load_all()

    # 2. NLP (GPU)
    # df = df.head(10000) # Decommenta per test veloci
    texts = (df['title'].astype(str) + ". " + df['overview'].astype(str)).tolist()
    embeddings = BertHandler().encode(texts)

    # 3. Core
    recsys = ContentBasedRecommender(df, embeddings)
    web = WebSearchService()
    trans = TranslationService()

    return df, recsys, web, trans, embeddings


with st.spinner("Avvio Sistema..."):
    df, recsys, web, trans, embeddings = init_backend()

render_main_page(df, recsys, web, trans, embeddings)