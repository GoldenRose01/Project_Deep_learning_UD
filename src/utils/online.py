from imdb import Cinemagoer


class OnlineMovieFetcher:
    def __init__(self):
        self.ia = Cinemagoer()

    def fetch_movie_data(self, title):
        """
        Cerca un film su IMDb e restituisce i dati formattati come il nostro DataFrame.
        """
        print(f"🌍 Sto cercando '{title}' su IMDb (Internet)...")
        try:
            # 1. Cerca il film
            search_results = self.ia.search_movie(title)

            if not search_results:
                return None

            # Prendi il primo risultato
            first_match = search_results[0]
            movie_id = first_match.movieID

            # 2. Scarica i dettagli completi (trama, generi)
            movie = self.ia.get_movie(movie_id)

            # 3. Estrai i dati
            real_title = movie.get('title', title)

            # Trama: a volte si chiama 'plot outline', a volte 'plot'
            overview = movie.get('plot outline')
            if not overview:
                plot_list = movie.get('plot')
                overview = plot_list[0] if plot_list else "Trama non disponibile."

            # Generi
            genres = movie.get('genres', [])
            genres_str = "|".join(genres)

            # Rating
            rating = movie.get('rating', 0.0)

            print(f"✅ Trovato online: {real_title}")

            # Restituisci un dizionario compatibile con il nostro DataFrame
            return {
                'title': real_title,
                'overview': overview,
                'genres': genres_str,
                'vote_average': rating,
                'is_online': True  # Flag per dire che viene da internet
            }

        except Exception as e:
            print(f"❌ Errore durante la ricerca online: {e}")
            return None