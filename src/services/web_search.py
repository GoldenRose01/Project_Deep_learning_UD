from imdb import Cinemagoer


class WebSearchService:
    def __init__(self):
        self.ia = Cinemagoer()

    def resolve_title(self, query):
        """Traduce 'Il Trono di Spade' in 'Game of Thrones'"""
        try:
            # Cerca film o serie
            res = self.ia.search_movie(query)
            if res:
                # Restituisce il titolo del PRIMO risultato (il più rilevante)
                return res[0]['title']
            return None
        except Exception as e:
            print(f"IMDb Error: {e}")
            return None

    def fetch_full_data(self, query):
        """Scarica metadati completi da IMDb"""
        try:
            res = self.ia.search_movie(query)
            if not res: return None

            # Prendi ID
            movie_id = res[0].movieID
            m = self.ia.get_movie(movie_id)

            # Trama (Gestione sicura errori)
            plot = "N/d"
            if 'plot outline' in m:
                plot = m['plot outline']
            elif 'plot' in m:
                plot = m['plot'][0]

            # Poster
            poster = m.get('full-size cover url', m.get('cover url'))

            return {
                'title': m.get('title'),
                'overview': plot,
                'genres': ", ".join(m.get('genres', [])),
                'vote_average': m.get('rating'),
                'source': 'IMDb Web',
                'type': 'Web Result',
                'custom_poster_url': poster
            }
        except Exception as e:
            print(f"Fetch Error: {e}")
            return None