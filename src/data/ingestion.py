import pandas as pd
import os
import glob
import kagglehub


class DataIngestor:
    def __init__(self):
        self.datasets = {
            'imdb_mov': "rounakbanik/the-movies-dataset",
            'netflix': "muhammadtahir194/netflix-movies-and-tv-shows-dataset",
            'imdb_tv': "devanshiipatel/imdb-tv-shows",
            'amazon': "shivamb/amazon-prime-movies-and-tv-shows",
            'rt': "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"
        }

    def _find_csv(self, path, pattern):
        files = glob.glob(os.path.join(path, "**", pattern), recursive=True)
        return files[0] if files else None

    def load_all(self):
        print("⬇️ SCARICAMENTO DATASETS...")
        frames = []

        # Iteriamo sui dataset e normalizziamo le colonne
        for key, slug in self.datasets.items():
            try:
                path = kagglehub.dataset_download(slug)

                if key == 'imdb_mov':
                    f = self._find_csv(path, "movies_metadata.csv")
                    df = pd.read_csv(f, low_memory=False)
                    df = df[['title', 'overview', 'genres', 'vote_average']].copy()
                    df['type'] = 'Movie';
                    df['source'] = 'IMDb Movie'
                    # Fix JSON genres
                    df['genres'] = df['genres'].astype(str).apply(lambda x: "|".join(
                        [y.split("'name': '")[1].split("'")[0] for y in x.split("},") if "'name': '" in y]))

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

                # Standardizzazione finale per questo chunk
                cols_needed = ['title', 'overview', 'genres', 'type', 'source', 'vote_average']
                available = [c for c in cols_needed if c in df.columns]
                frames.append(df[available])

            except Exception as e:
                print(f"⚠️ Errore con {key}: {e}")

        print("🔗 Unione Dataset...")
        df_final = pd.concat(frames, ignore_index=True)

        # Pulizia globale
        df_final['overview'] = df_final['overview'].fillna("")
        df_final = df_final[df_final['overview'].str.len() > 20]
        df_final = df_final.drop_duplicates(subset=['title'])
        df_final['vote_average'] = pd.to_numeric(df_final['vote_average'], errors='coerce').fillna(0)

        return df_final