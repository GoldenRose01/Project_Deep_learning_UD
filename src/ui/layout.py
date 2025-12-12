import streamlit as st
from src.ui.components import render_movie_card
from src.ml.benchmark import BenchmarkRunner


def render_main_page(df, recsys, web_search, translator, embeddings):
    st.title("🚀 AI Movie System Enterprise")

    # Sidebar
    lang = st.sidebar.selectbox("Lingua", ["en", "it", "es"])

    # Tabs
    t1, t2, t3 = st.tabs(["🔎 Ricerca", "🧠 Laboratorio AI", "📊 Statistiche"])

    with t1:
        query = st.text_input("Cerca Film (anche in Italiano)")
        if st.button("Vai") and query:
            # 1. Risoluzione Titolo
            resolved = web_search.resolve_title(query)
            target = resolved if resolved else query
            st.info(f"Target identificato: **{target}**")

            # 2. Ricerca Locale
            recs = recsys.recommend_single(target)

            if recs is not None:
                st.success("Trovato nel DB Locale!")
                for _, row in recs.iterrows():
                    render_movie_card(row, translator, lang)
            else:
                st.warning("Non in DB. Scarico dal Web...")
                web_data = web_search.fetch_full_data(target)
                if web_data:
                    render_movie_card(web_data, translator, lang)
                else:
                    st.error("Non trovato.")

    with t2:
        if st.button("Avvia Benchmark Modelli"):
            y = df['source']
            mask = y.isin(y.value_counts()[y.value_counts() > 100].index)
            runner = BenchmarkRunner(embeddings[mask], y[mask])
            st.dataframe(runner.run())