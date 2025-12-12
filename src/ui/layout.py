import streamlit as st
from src.ui.components import render_movie_card
from src.ml.benchmark import BenchmarkRunner


def render_main_page(df, recsys, web_search, translator, embeddings):
    st.title("🚀 AI Movie System Enterprise")

    # Prepara la lista titoli per l'autocomplete (ordinata)
    all_titles = sorted(df['title'].astype(str).unique())

    # Sidebar
    lang = st.sidebar.selectbox("Lingua / Language", ["en", "it", "es"])

    # Tabs
    t1, t2, t3, t4 = st.tabs(["🔎 Smart Search", "👤 Crea Profilo", "🧠 Lab AI", "📊 Stats"])

    # --- TAB 1: RICERCA SINGOLA (IBRIDA) ---
    with t1:
        st.subheader("Ricerca Universale")
        st.info("Cerca nel Database locale o su Internet (se non lo abbiamo).")

        # 1. Input Utente
        col_search_1, col_search_2 = st.columns([3, 1])
        with col_search_1:
            query = st.text_input("Scrivi un titolo (es. Matrix, Trono di Spade)",
                                  placeholder="Premi Invio per cercare...")
        with col_search_2:
            force_web = st.checkbox("Forza Web Search")

        if query:
            # A. TENTATIVO LOCALE (FUZZY)
            # Cerchiamo se la stringa è contenuta in qualche titolo locale
            matches = df[df['title'].str.contains(query, case=False, regex=False)]

            local_found = False

            if not matches.empty and not force_web:
                st.success(f"Trovati {len(matches)} risultati locali.")
                # Selettore per disambiguare (es. "Matrix" -> "The Matrix", "Matrix Reloaded")
                selected_local = st.selectbox("Seleziona il film esatto:", matches['title'].tolist())

                if selected_local:
                    recs = recsys.recommend_single(selected_local)
                    if recs is not None:
                        st.markdown(f"### Simili a: **{selected_local}**")
                        for _, row in recs.iterrows():
                            render_movie_card(row, translator, lang)
                        local_found = True

            # B. TENTATIVO WEB (Se non trovato locale o forzato)
            if matches.empty or force_web:
                if not local_found: st.warning("Non trovato esattamente nel DB locale. Cerco su Internet...")

                # Risoluzione Titolo (es. Trono di Spade -> Game of Thrones)
                with st.spinner(f"Interrogazione IMDb per '{query}'..."):
                    resolved_title = web_search.resolve_title(query)

                if resolved_title:
                    st.info(f"Titolo Internazionale Rilevato: **{resolved_title}**")

                    # Riprova ricerca locale con titolo risolto
                    matches_resolved = df[df['title'].str.contains(resolved_title, case=False, regex=False)]
                    if not matches_resolved.empty and not force_web:
                        st.success(f"Ah! Con il titolo inglese '{resolved_title}' l'abbiamo trovato nel DB!")
                        recs = recsys.recommend_single(matches_resolved.iloc[0]['title'])
                        for _, row in recs.iterrows():
                            render_movie_card(row, translator, lang)
                    else:
                        # Scarica dati completi dal web
                        web_data = web_search.fetch_full_data(resolved_title)
                        if web_data:
                            st.markdown("### Risultato Web")
                            render_movie_card(web_data, translator, lang)
                        else:
                            st.error("Nessun risultato trovato neanche online.")
                else:
                    st.error("Titolo non riconosciuto da IMDb.")

    # --- TAB 2: PROFILO UTENTE (AUTOCOMPLETE) ---
    with t2:
        st.subheader("Generatore di Profilo")
        st.write("Seleziona i film che hai amato. L'AI calcolerà la media dei tuoi gusti.")

        # AUTOCOMPLETE REALE (Multiselect)
        selected_movies = st.multiselect(
            "Aggiungi film al tuo profilo:",
            options=all_titles,
            placeholder="Scrivi per cercare (es. The Matrix)..."
        )

        if st.button("Genera Consigli Personalizzati") and selected_movies:
            with st.spinner("Calcolo vettore medio del profilo..."):
                recs_profile = recsys.recommend_profile(selected_movies)

            if recs_profile is not None:
                st.balloons()
                st.markdown("### Consigliati per il tuo mix unico:")
                for _, row in recs_profile.iterrows():
                    render_movie_card(row, translator, lang)
            else:
                st.error("Errore nel calcolo del profilo.")

    # --- TAB 3: AI LAB ---
    with t3:
        st.write("Benchmark Modelli")
        if st.button("Avvia Gara Neural Networks"):
            y = df['source']
            # Filtro per avere abbastanza dati per classe
            valid_sources = y.value_counts()[y.value_counts() > 50].index
            mask = y.isin(valid_sources)
            runner = BenchmarkRunner(embeddings[mask], y[mask])
            st.dataframe(runner.run())

    with t4:
        st.metric("Film Totali nel DB", len(df))
        st.metric("Sorgenti", ", ".join(df['source'].unique()))