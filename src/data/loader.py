import pandas as pd
import os
import glob
import kagglehub


class DataLoader:
    def __init__(self):
        self.paths = {}

    def download_data(self):
        print("⬇️  SCARICAMENTO DATASETS (Potrebbe richiedere tempo)...")
        # 1. IMDb Movies
        self.paths['imdb_mov'] = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
        # 2. Netflix
        self.paths['netflix'] = kagglehub.dataset_download("muhammadtahir194/netflix-movies-and-tv-shows-dataset")
        # 3. IMDb TV
        self.paths['imdb_tv'] = kagglehub.dataset_download("devanshiipatel/imdb-tv-shows")
        # 4. Amazon Prime (NUOVO)
        self.paths['amazon'] = kagglehub.dataset_download("shivamb/amazon-prime-movies-and-tv-shows")
        # 5. Rotten Tomatoes (NUOVO)
        self.paths['rt'] = kagglehub.dataset_download(
            "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset")
        # 6. MovieLens 25M (NUOVO)
        self.paths['movielens'] = kagglehub.dataset_download("grouplens/movielens-20m-dataset")  # O 25m

    def _find_file(self, folder_key, filename_pattern):
        """Trova un file csv in modo ricorsivo nella cartella scaricata"""
        path = self.paths.get(folder_key)
        if not path: return None
        files = glob.glob(os.path.join(path, "**", filename_pattern), recursive=True)
        return files[0] if files else None

    def load_merged_data(self):
        self.download_data()
        frames = []

        # --- 1. NETFLIX ---
        try:
            f = self._find_file('netflix', "netflix_titles.csv")
            df = pd.read_csv(f)
            df = df.rename(columns={'description': 'overview', 'listed_in': 'genres'})
            df['source'] = 'Netflix'
            df['vote_average'] = 0
            # Netflix ha già la colonna type
            frames.append(df[['title', 'overview', 'genres', 'type', 'source', 'vote_average']])
        except:
            print("⚠️ Skip Netflix")

        # --- 2. AMAZON PRIME ---
        try:
            f = self._find_file('amazon', "amazon_prime_titles.csv")
            df = pd.read_csv(f)
            df = df.rename(columns={'description': 'overview', 'listed_in': 'genres'})
            df['source'] = 'Amazon Prime'
            df['vote_average'] = 0
            frames.append(df[['title', 'overview', 'genres', 'type', 'source', 'vote_average']])
        except:
            print("⚠️ Skip Amazon")

        # --- 3. ROTTEN TOMATOES ---
        try:
            f = self._find_file('rt', "rotten_tomatoes_movies.csv")
            df = pd.read_csv(f)
            # RT ha 'movie_info' come trama
            df = df.rename(columns={'movie_title': 'title', 'movie_info': 'overview'})
            df['source'] = 'Rotten Tomatoes'
            df['type'] = 'Movie'
            df['vote_average'] = df.get('tomatometer_rating', 0)
            frames.append(df[['title', 'overview', 'genres', 'type', 'source', 'vote_average']])
        except:
            print("⚠️ Skip Rotten Tomatoes")

        # --- 4. IMDb MOVIES ---
        try:
            f = self._find_file('imdb_mov', "movies_metadata.csv")
            df = pd.read_csv(f, low_memory=False)
            df = df[['title', 'overview', 'genres', 'vote_average']].copy()
            df['type'] = 'Movie'
            df['source'] = 'IMDb Movie'
            # Pulizia generi JSON
            df['genres'] = df['genres'].astype(str).apply(
                lambda x: "|".join([y.split("'name': '")[1].split("'")[0] for y in x.split("},") if "'name': '" in y]))
            frames.append(df)
        except:
            print("⚠️ Skip IMDb Movies")

        # --- 5. IMDb TV SHOWS ---
        try:
            f = self._find_file('imdb_tv', "*.csv")  # Nome variabile
            df = pd.read_csv(f)
            # Mappa dinamica colonne
            cols = {c.lower(): c for c in df.columns}
            df = df.rename(columns={
                cols.get('title', 'Title'): 'title',
                cols.get('description', 'Description'): 'overview',
                cols.get('genre', 'Genre'): 'genres',
                cols.get('rating', 'Rating'): 'vote_average'
            })
            if 'overview' not in df.columns and 'summary' in cols:
                df = df.rename(columns={cols['summary']: 'overview'})

            df['source'] = 'IMDb TV'
            df['type'] = 'TV Show'
            available = [c for c in ['title', 'overview', 'genres', 'type', 'source', 'vote_average'] if
                         c in df.columns]
            frames.append(df[available])
        except:
            print("⚠️ Skip IMDb TV")

        # --- 6. MOVIELENS (Solo Metadata, niente trame lunghe di solito) ---
        # Nota: MovieLens base NON ha trame. Usiamo solo i titoli per ampliare il catalogo
        # o lo saltiamo per l'embedding se non abbiamo la trama.
        # Per ora lo saltiamo per non sporcare il modello semantico con trame vuote.

        # --- MERGE FINALE ---
        print("🔗 Unione dei dataset...")
        df_final = pd.concat(frames, ignore_index=True)

        # Pulizia
        df_final['overview'] = df_final['overview'].fillna("")
        df_final = df_final[df_final['overview'].str.len() > 20]  # Rimuove film senza trama
        df_final = df_final.drop_duplicates(subset=['title'])  # Rimuove doppioni

        # Gestione Poster (Nessuno di questi dataset nuovi ha poster URL affidabili nel CSV)
        df_final['poster_path'] = None

        print(f"✅ DATASET DEFINITIVO: {len(df_final)} titoli unici caricati.")
        return df_final