import streamlit as st
import pandas as pd
import os
import shutil
from src.ui.components import render_movie_card
from src.ml.benchmark import BenchmarkRunner


def render_main_page(df, recsys, web_search, translator, embeddings):
    st.title("🚀 AI Movie System Enterprise")

    all_titles = sorted(df['title'].unique().tolist())
    lang = st.sidebar.selectbox("Lingua / Language", ["en", "it", "es"])

    # 5 Tab
    t1, t2, t3, t4, t5 = st.tabs(["🔎 Ricerca", "👤 Profilo", "🧠 Lab AI", "📊 Stats", "💾 Dataset Custom"])

    # --- TAB 1: RICERCA ---
    with t1:
        st.subheader("Trova un film specifico")
        selected_titles = st.multiselect("Cerca nel Database:", options=all_titles, max_selections=1,
                                         placeholder="Es. Matrix...")

        if selected_titles:
            target = selected_titles[0]
            st.success(f"Analisi per: **{target}**")
            recs = recsys.recommend_single(target)
            if recs is not None:
                for _, row in recs.iterrows():
                    render_movie_card(row, translator, lang)

        st.markdown("---")
        with st.expander("Non trovi il film? Cerca Online"):
            web_query = st.text_input("Scrivi titolo per ricerca Web", key="web_q")
            if st.button("Cerca Web") and web_query:
                with st.spinner(f"Cerco '{web_query}' su IMDb..."):
                    resolved = web_search.resolve_title(web_query)
                    if resolved and resolved in all_titles:
                        st.success(f"Trovato nel DB: **{resolved}**! Selezionalo sopra.")
                    else:
                        data = web_search.fetch_full_data(resolved if resolved else web_query)
                        if data:
                            render_movie_card(data, translator, lang)
                        else:
                            st.error("Nessun risultato.")

    # --- TAB 2: PROFILO ---
    with t2:
        st.subheader("Profilo Misto")
        profile_movies = st.multiselect("I tuoi preferiti:", options=all_titles, max_selections=10)
        if st.button("Genera Mix") and profile_movies:
            with st.spinner("Calcolo media vettoriale..."):
                recs_profile = recsys.recommend_profile(profile_movies)
            if recs_profile is not None:
                st.balloons()
                for _, row in recs_profile.iterrows():
                    render_movie_card(row, translator, lang)

    # --- TAB 3: AI LAB ---
    with t3:
        if st.button("Avvia Benchmark"):
            y = df['source']
            valid_sources = y.value_counts()[y.value_counts() > 50].index
            mask = y.isin(valid_sources)
            runner = BenchmarkRunner(embeddings[mask], y[mask])
            st.dataframe(runner.run())

    # --- TAB 4: STATS ---
    with t4:
        st.metric("Film Totali", len(df))
        st.bar_chart(df['source'].value_counts())

    # --- TAB 5: DATASET CUSTOM (NUOVO) ---
    with t5:
        st.header("Importa i tuoi Dati")
        st.info("Carica un file CSV. Il sistema lo integrerà nel database per la ricerca e l'AI.")

        uploaded_file = st.file_uploader("Carica CSV", type=['csv'])

        if uploaded_file:
            try:
                # Anteprima
                preview_df = pd.read_csv(uploaded_file)
                st.write("Anteprima dati:", preview_df.head(3))

                st.subheader("Mappatura Colonne")
                cols = preview_df.columns.tolist()

                c1, c2, c3 = st.columns(3)
                col_title = c1.selectbox("Colonna Titolo", cols, index=0)
                col_plot = c2.selectbox("Colonna Trama/Descrizione", cols, index=1 if len(cols) > 1 else 0)
                col_genre = c3.selectbox("Colonna Genere (Opzionale)", ["Nessuna"] + cols)

                if st.button("💾 Salva e Integra nel Sistema", type="primary"):
                    # Normalizzazione
                    clean_df = pd.DataFrame()
                    clean_df['title'] = preview_df[col_title]
                    clean_df['overview'] = preview_df[col_plot]

                    if col_genre != "Nessuna":
                        clean_df['genres'] = preview_df[col_genre]
                    else:
                        clean_df['genres'] = "Custom"

                    clean_df['vote_average'] = 0

                    # Salvataggio
                    save_path = os.path.join("custom_datasets", uploaded_file.name)
                    clean_df.to_csv(save_path, index=False)

                    st.success(f"Salvato in {save_path}!")

                    # PULIZIA CACHE PER FORZARE RICARICAMENTO
                    if os.path.exists("cache/movies_data.pkl"):
                        os.remove("cache/movies_data.pkl")

                    # Pulsante magico per riavviare
                    st.warning("⚠️ Cache pulita. Ricarica la pagina (F5) o clicca Rerun per processare i nuovi dati.")
                    if st.button("Riavvia Sistema Ora"):
                        st.rerun()

            except Exception as e:
                st.error(f"Errore lettura CSV: {e}")

        # Mostra file già caricati
        st.markdown("---")
        st.subheader("File Custom Attivi")
        if os.path.exists("custom_datasets"):
            files = os.listdir("custom_datasets")
            if files:
                for f in files:
                    st.text(f"📄 {f}")
                    if st.button(f"🗑️ Elimina {f}"):
                        os.remove(os.path.join("custom_datasets", f))
                        os.remove("cache/movies_data.pkl")
                        st.rerun()
            else:
                st.caption("Nessun file custom caricato.")