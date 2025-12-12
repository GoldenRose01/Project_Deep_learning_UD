import pandas as pd
import os
import glob
import kagglehub


class DataIngestor:
    def __init__(self):
        # Percorso dove salveremo i dati processati
        self.cache_path = os.path.join("cache", "movies_data.pkl")

        self.datasets = {
            'imdb_mov': "rounakbanik/the-movies-dataset",
            'netflix': "muhammadtahir194/netflix-movies-and-tv-shows-dataset",
            'imdb_tv': "devanshiipatel/imdb-tv-shows",
            'amazon': "shivamb/amazon-prime-movies-and-tv-shows",
            'rt': "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset",
            'disney': "shivamb/disney-movies-and-tv-shows",
            'hulu': "shivamb/hulu-movies-and-tv-shows",
            'wiki': "jrobischon/wikipedia-movie-plots"  # <--- NUOVO MASSIVO
        }

    def _find_csv(self, path, pattern):
        files = glob.glob(os.path.join(path, "**", pattern), recursive=True)
        return files[0] if files else None

    def load_all(self):
        # 1. CONTROLLO CACHE
        if os.path.exists(self.cache_path):
            print(f"⚡ CACHE TROVATA: Carico dati da {self.cache_path}...")
            return pd.read_pickle(self.cache_path)

        print("⬇️ NESSUNA CACHE. SCARICAMENTO DATASETS...")
        frames = []

        for key, slug in self.datasets.items():
            try:
                path = kagglehub.dataset_download(slug)
                df = None

                # --- LOGICA DI PARSING ESISTENTE ---
                if key == 'imdb_mov':
                    f = self._find_csv(path, "movies_metadata.csv")
                    df = pd.read_csv(f, low_memory=False)
                    df = df[['title', 'overview', 'genres', 'vote_average']].copy()
                    df['type'] = 'Movie';
                    df['source'] = 'IMDb Movie'
                    df['genres'] = df['genres'].astype(str).apply(lambda x: "|".join(
                        [y.split("'name': '")[1].split("'")[0] for y in x.split("},") if "'name': '" in y]))

                elif key == 'wiki':  # --- NUOVO PARSING WIKIPEDIA ---
                    f = self._find_csv(path, "wiki_movie_plots_deduped.csv")
                    df = pd.read_csv(f)
                    df = df.rename(columns={'Title': 'title', 'Plot': 'overview', 'Genre': 'genres'})
                    df['source'] = 'WikiArchive';
                    df['type'] = 'Movie';
                    df['vote_average'] = 0

                elif key == 'netflix':
                    f = self._find_csv(path, "netflix_titles.csv")
                    df = pd.read_csv(f)
                    df = df.rename(columns={'description': 'overview', 'listed_in': 'genres'})
                    df['source'] = 'Netflix';
                    df['vote_average'] = 0

                elif key == 'imdb_tv':
                    f = self._find_csv(path, "*.csv")
                    df = pd.read_csv(f)
                    cols = {c.lower(): c for c in df.columns}
                    df = df.rename(columns={cols.get('title', 'Title'): 'title',
                                            cols.get('description', 'Description'): 'overview',
                                            cols.get('genre', 'Genre'): 'genres',
                                            cols.get('rating', 'Rating'): 'vote_average'})
                    if 'overview' not in df.columns and 'summary' in cols: df = df.rename(
                        columns={cols['summary']: 'overview'})
                    df['type'] = 'TV Show';
                    df['source'] = 'IMDb TV'

                elif key == 'amazon':
                    f = self._find_csv(path, "amazon_prime_titles.csv")
                    df = pd.read_csv(f)
                    df = df.rename(columns={'description': 'overview', 'listed_in': 'genres'})
                    df['source'] = 'Amazon';
                    df['vote_average'] = 0

                elif key == 'rt':
                    f = self._find_csv(path, "rotten_tomatoes_movies.csv")
                    df = pd.read_csv(f)
                    df = df.rename(columns={'movie_title': 'title', 'movie_info': 'overview',
                                            'tomatometer_rating': 'vote_average'})
                    df['source'] = 'Rotten Tomatoes';
                    df['type'] = 'Movie'

                elif key == 'disney':
                    f = self._find_csv(path, "disney_plus_titles.csv")
                    df = pd.read_csv(f)
                    df = df.rename(columns={'description': 'overview', 'listed_in': 'genres'})
                    df['source'] = 'Disney+';
                    df['vote_average'] = 0

                elif key == 'hulu':
                    f = self._find_csv(path, "hulu_titles.csv")
                    df = pd.read_csv(f)
                    df = df.rename(columns={'description': 'overview', 'listed_in': 'genres'})
                    df['source'] = 'Hulu';
                    df['vote_average'] = 0

                if df is not None:
                    cols = ['title', 'overview', 'genres', 'type', 'source', 'vote_average']
                    available = [c for c in cols if c in df.columns]
                    frames.append(df[available])
                    print(f"✅ OK: {key}")

            except Exception as e:
                print(f"⚠️ Errore {key}: {e}")

        print("🔗 Unione Dataset...")
        df_final = pd.concat(frames, ignore_index=True)

        # Pulizia
        df_final['overview'] = df_final['overview'].fillna("")
        df_final = df_final[df_final['overview'].str.len() > 20]
        df_final = df_final.drop_duplicates(subset=['title'])
        df_final['vote_average'] = pd.to_numeric(df_final['vote_average'], errors='coerce').fillna(0)

        # 2. SALVATAGGIO CACHE
        print(f"💾 SALVATAGGIO CACHE IN {self.cache_path}...")
        df_final.to_pickle(self.cache_path)

        print(f"🎉 TOTALE: {len(df_final)} titoli.")
        return df_final