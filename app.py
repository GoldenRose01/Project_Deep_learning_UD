import streamlit as st
import pandas as pd
from src.data.loader import DataLoader
from src.models.embeddings import EmbeddingGenerator
from src.models.recommender import MovieRecommender
from src.models.classifier import GenreClassifier
from src.utils.online import OnlineMovieFetcher
from src.utils.ui_helpers import render_movie_card  # La nostra nuova funzione grafica

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Movie AI Lab", page_icon="🧪", layout="wide")

# --- DIZIONARIO TESTI ---
trans = {
    "it": {"title": "🎬 AI Movie Lab", "sim": "Affinità", "plot": "Trama", "translating": "Traduzione...",
           "rec_header": "Basato sui tuoi gusti, ti consigliamo:"},
    "en": {"title": "🎬 AI Movie Lab", "sim": "Similarity", "plot": "Plot", "translating": "Translating...",
           "rec_header": "Based on your taste, we recommend:"},
    "es": {"title": "🎬 AI Movie Lab", "sim": "Similitud", "plot": "Trama", "translating": "Traduciendo...",
           "rec_header": "Basado en tus gustos, recomendamos:"}
}


# --- CARICAMENTO RISORSE ---
@st.cache_resource
def load_app_resources():
    loader = DataLoader()
    df = loader.load_merged_data()

    # ⚠️ LIMITATORE DI RAM ⚠️
    # Con 6 dataset superiamo i 100k film. BERT ci mette troppo.
    # Prendiamo i primi 100000 (o mescoliamo)
    if len(df) > 100000:
        # Ordiniamo per popolarità/voti se possibile, altrimenti random sample
        # Qui facciamo un shuffle per avere un mix di tutto
        df = df.sample(100000, random_state=42).reset_index(drop=True)

    embedder = EmbeddingGenerator(method='bert')
    df['combined_text'] = "Title: " + df['title'].astype(str) + ". Plot: " + df['overview'].astype(str)

    # Questo è il passaggio lento (qualche minuto la prima volta)
    embeddings = embedder.fit_transform(df['combined_text'].tolist())

    recsys = MovieRecommender(df, embeddings)
    all_titles = sorted(df['title'].astype(str).unique().tolist())

    return df, embeddings, recsys, all_titles


# --- SIDEBAR ---
st.sidebar.title("⚙️ Control Panel")
lang_code = st.sidebar.selectbox("Lingua / Language", ["it", "en", "es"])
t = trans[lang_code]

with st.spinner("Inizializzazione Sistema AI (Caricamento dati)..."):
    df, embeddings, recsys, all_titles = load_app_resources()

# --- TABS ---
tab_rec, tab_train = st.tabs(["🍿 Profilo & Raccomandazioni", "🧠 Training AI"])

# ==========================================
# TAB 1: RACCOMANDAZIONI AVANZATE
# ==========================================
with tab_rec:
    col_sx, col_dx = st.columns([1, 2])

    with col_sx:
        st.subheader("1. Crea il tuo Profilo")
        st.info("Seleziona fino a 10 titoli che ti piacciono. L'AI capirà i tuoi gusti calcolando la media semantica.")

        # MULTI-SELECT (Cuore della nuova funzionalità)
        selected_movies = st.multiselect(
            "Cosa hai visto e ti è piaciuto?",
            options=all_titles,
            max_selections=10,
            placeholder="Scrivi titoli (es. Matrix, Inception...)"
        )

        st.subheader("2. Filtri")
        # FILTRO TIPO
        filter_type = st.radio(
            "Cosa vuoi vedere stasera?",
            ["All", "Movies Only", "TV Shows Only"],
            horizontal=True
        )

        # BOTTONE
        btn_recommend = st.button("✨ Genera Consigli", type="primary")

    with col_dx:
        if btn_recommend and selected_movies:
            st.subheader(t["rec_header"])

            # Chiamata al nuovo metodo del Recommender
            results, msg = recsys.get_profile_recommendations(
                selected_movies,
                filter_type=filter_type,
                top_n=5
            )

            if results is not None and not results.empty:
                for idx, row in results.iterrows():
                    render_movie_card(row, lang_code, t)
            else:
                st.warning(msg)
        elif btn_recommend and not selected_movies:
            st.error("Per favore seleziona almeno un film per creare il tuo profilo.")
        else:
            # Stato iniziale vuoto
            st.markdown("""
            <div style="text-align: center; margin-top: 50px; color: gray;">
                <h3>👈 Seleziona i tuoi film preferiti a sinistra per iniziare!</h3>
                <p>L'algoritmo creerà un vettore unico che rappresenta la tua personalità cinematografica.</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 2: TRAINING (Codice Classificazione)
# ==========================================
with tab_train:
    st.header("Laboratorio Deep Learning 🧠")

    c1, c2 = st.columns(2)
    target_col = c1.selectbox("Target Predizione", ["type", "source", "genres"])
    epochs = c2.slider("Epoche", 1, 50, 10)

    if st.button("🚀 Start Training"):
        with st.spinner("Training..."):
            y = df[target_col]
            if target_col == 'genres':  # Semplificazione generi
                y = df['genres'].astype(str).apply(lambda x: x.split('|')[0] if '|' in x else x)

            # Filtro classi rare
            counts = y.value_counts()
            valid_classes = counts[counts > 20].index  # Almeno 20 esempi
            mask = y.isin(valid_classes)

            classifier = GenreClassifier(embeddings[mask], y[mask])
            classifier.train(epochs=epochs)
            acc, report = classifier.evaluate()

            st.balloons()
            st.success(f"Accuratezza: **{acc:.2%}**")
            st.dataframe(pd.DataFrame(report).transpose().style.highlight_max(axis=0), use_container_width=True)