import pandas as pd
import streamlit as st
from deep_translator import GoogleTranslator


@st.cache_data
def translate_text(text, target_lang):
    if target_lang == 'en' or pd.isna(text) or text == "": return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text[:2000])
    except:
        return text


def render_movie_card(row, lang_code, translations):
    """Disegna la card del film nella UI"""
    with st.container():
        col_img, col_txt = st.columns([1, 4])

        with col_img:
            # Usiamo un placeholder carino se manca l'immagine
            st.image("https://via.placeholder.com/150x225?text=No+Poster", width="stretch")
            st.metric(translations["sim"], f"{float(row['score']):.2%}")

        with col_txt:
            # --- LOGICA COLORI BADGE ---
            src = row.get('source', 'Unknown')

            if 'Netflix' in src:
                color = 'red'
            elif 'Amazon' in src:
                color = 'blue'
            elif 'Rotten' in src:
                color = 'orange'  # Tomato color
            elif 'IMDb TV' in src:
                color = 'violet'
            else:
                color = 'green'  # IMDb Movie classico

            type_badge = "📺 TV Show" if row.get('type') == 'TV Show' else "🎬 Movie"

            # Titolo e Badge
            st.markdown(f"### {row['title']}")
            st.markdown(f":{color}[**{src}**] | **{type_badge}** | ⭐ {row.get('vote_average', 0)}")
            st.caption(f"Genre: {row.get('genres', '')}")

            # Traduzione
            plot_text = str(row['overview'])
            if lang_code != 'en':
                with st.spinner(translations['translating']):
                    plot_text = translate_text(plot_text, lang_code)

            st.markdown(f"**{translations['plot']}:**")
            st.markdown(f"<div style='text-align: justify;'>{plot_text}</div>", unsafe_allow_html=True)
    st.divider()