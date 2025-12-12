import streamlit as st


def render_movie_card(row, translation_service=None, target_lang='en'):
    with st.container():
        c1, c2 = st.columns([1, 4])

        # Immagine
        url = row.get('custom_poster_url')
        if not url:
            path = row.get('poster_path')
            url = f"https://image.tmdb.org/t/p/w300{path}" if path else "https://via.placeholder.com/150"

        with c1:
            st.image(url, width=150)

        with c2:
            st.markdown(f"### {row['title']}")

            # Badge Colore
            src = row.get('source', 'Unknown')
            color = 'red' if 'Netflix' in src else 'blue' if 'Amazon' in src else 'green'
            st.markdown(f":{color}[**{src}**] | ⭐ {row.get('vote_average', 'N/d')}")

            # Trama tradotta
            plot = str(row['overview'])
            if translation_service and target_lang != 'en':
                plot = translation_service.translate_from_en(plot, target_lang)
            st.write(plot)

            if 'score' in row: st.caption(f"Similarity: {row['score']:.1%}")
    st.divider()