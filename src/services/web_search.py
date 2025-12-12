from imdb import Cinemagoer


class WebSearchService:
    def __init__(self):
        self.ia = Cinemagoer()

    def resolve_title(self, query):
        """Traduce 'Il Trono di Spade' in 'Game of Thrones'"""
        try:
            res = self.ia.search_movie(query)
            return res[0]['title'] if res else None
        except:
            return None

    def fetch_full_data(self, query):
        """Scarica metadati completi da IMDb"""
        try:
            res = self.ia.search_movie(query)
            if not res: return None

            m = self.ia.get_movie(res[0].movieID)
            return {
                'title': m.get('title'),
                'overview': m.get('plot outline') or m.get('plot', ['N/d'])[0],
                'genres': ", ".join(m.get('genres', [])),
                'vote_average': m.get('rating'),
                'source': 'IMDb Web',
                'type': 'Web Result',
                'custom_poster_url': m.get('full-size cover url', m.get('cover url'))
            }
        except:
            return None