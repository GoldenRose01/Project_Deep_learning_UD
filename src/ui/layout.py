import streamlit as st
from src.ui.components import render_movie_card
from src.ml.benchmark import BenchmarkRunner


def render_main_page(df, recsys, web_search, translator, embeddings):
    st.title("🚀 AI Movie System Enterprise")

    # Prepara la lista completa dei titoli per l'autocomplete
    # (È veloce perché è solo una lista di stringhe in memoria)
    all_titles = sorted(df['title'].unique().tolist())

    # Sidebar
    lang = st.sidebar.selectbox("Lingua / Language", ["en", "it", "es"])

    # Tabs riorganizzate
    t1, t2, t3, t4 = st.tabs(["🔎 Ricerca Rapida", "👤 Profilo Misto (Max 10)", "🧠 Lab AI", "📊 Stats"])

    # --- TAB 1: RICERCA SINGOLA (AUTOCOMPLETE) ---
    with t1:
        st.subheader("Trova un film specifico")
        st.caption("Scrivi parte del titolo e seleziona quello corretto dalla lista.")

        # 1. AUTOCOMPLETE LOCALE
        # Usiamo multiselect limitato a 1 per avere l'effetto "cerca e seleziona" con cancellazione facile
        selected_titles = st.multiselect(
            "Cerca nel Database:",
            options=all_titles,
            max_selections=1,
            placeholder="Es. Matrix, Avatar, Godfather..."
        )

        # Se l'utente ha selezionato qualcosa dal menu a tendina
        if selected_titles:
            target = selected_titles[0]
            st.success(f"Analisi per: **{target}**")

            # Raccomandazione Immediata
            recs = recsys.recommend_single(target)
            if recs is not None:
                st.markdown("### 🎬 Film Simili")
                for _, row in recs.iterrows():
                    render_movie_card(row, translator, lang)

        # 2. FALLBACK ONLINE (Se non trova nel menu a tendina)
        st.markdown("---")
        with st.expander("Non trovi il film nella lista sopra? Cerca Online"):
            st.warning("Usa questo solo se il film non appare nell'autocomplete sopra.")
            col_web_1, col_web_2 = st.columns([3, 1])
            with col_web_1:
                web_query = st.text_input("Scrivi titolo per ricerca Web", key="web_q")
            with col_web_2:
                btn_web = st.button("Cerca Web")

            if btn_web and web_query:
                # Risoluzione titolo e scaricamento
                with st.spinner(f"Cerco '{web_query}' su IMDb..."):
                    resolved = web_search.resolve_title(web_query)

                    if resolved and resolved in all_titles:
                        st.success(f"Trovato! Il titolo corretto è **{resolved}** (è presente nel DB locale).")
                        st.info("Per favore selezionalo dal menu in alto per vedere i simili.")
                    else:
                        # Scarica dati
                        data = web_search.fetch_full_data(resolved if resolved else web_query)
                        if data:
                            st.markdown("### Risultato Web")
                            render_movie_card(data, translator, lang)
                        else:
                            st.error("Nessun risultato trovato.")

    # --- TAB 2: PROFILO MULTIPLO ---
    with t2:
        st.subheader("Costruisci il tuo Profilo")
        st.write("Seleziona da 2 a 10 film per creare un mix dei tuoi gusti.")

        # SELETTORE MULTIPLO
        profile_movies = st.multiselect(
            "I tuoi film preferiti:",
            options=all_titles,
            max_selections=10,
            placeholder="Aggiungi film alla lista..."
        )

        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            btn_profile = st.button("✨ Genera Mix", type="primary")

        if btn_profile:
            if not profile_movies:
                st.error("Seleziona almeno un film!")
            else:
                with st.spinner(f"Sto calcolando la media vettoriale di {len(profile_movies)} film..."):
                    # Chiama la funzione per il profilo
                    recs_profile = recsys.recommend_profile(profile_movies, top_n=10)

                if recs_profile is not None:
                    st.balloons()
                    st.success("Ecco i film che si trovano nell'intersezione dei tuoi gusti:")
                    for _, row in recs_profile.iterrows():
                        render_movie_card(row, translator, lang)
                else:
                    st.error("Impossibile generare consigli. Riprova con titoli diversi.")

    # --- TAB 3: BENCHMARK ---
    with t3:
        st.header("Benchmark Modelli")
        if st.button("Avvia Gara Neural Networks"):
            y = df['source']
            valid_sources = y.value_counts()[y.value_counts() > 50].index
            mask = y.isin(valid_sources)
            runner = BenchmarkRunner(embeddings[mask], y[mask])
            st.dataframe(runner.run())

    # --- TAB 4: STATISTICHE ---
    with t4:
        st.metric("Film Totali nel DB", len(df))
        st.markdown("### Distribuzione Sorgenti")
        st.bar_chart(df['source'].value_counts())