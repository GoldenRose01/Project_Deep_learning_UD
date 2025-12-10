import pandas as pd
import os

class DataLoader:
    def __init__(self, filepath='movies.csv'):
        self.filepath = filepath

    def load_data(self):
        """
        Carica il dataset. Se il file non esiste, crea un dataset dummy
        per permettere il testing immediato.
        """
        if not os.path.exists(self.filepath):
            print(f"⚠️  File '{self.filepath}' non trovato. Creazione dataset di TEST...")
            return self._create_dummy_dataset()
        
        print(f"📂 Caricamento dati da {self.filepath}...")
        df = pd.read_csv(self.filepath)
        
        # Pulizia base: ci servono solo titolo, trama e generi
        # Assumiamo che il CSV abbia colonne simili a IMDB dataset
        required_cols = ['title', 'overview', 'genres']
        
        # Controllo se le colonne esistono (adattamento nomi per dataset diversi)
        df.columns = [c.lower() for c in df.columns] # tutto minuscolo per sicurezza
        
        # Filtriamo righe vuote
        df = df.dropna(subset=['overview', 'title'])
        return df

    def _create_dummy_dataset(self):
        """Crea un piccolo dataset in memoria per testare il codice."""
        data = {
            'title': [
                'Toy Story', 
                'Finding Nemo', 
                'The Godfather', 
                'Pulp Fiction', 
                'Star Wars: A New Hope',
                'Interstellar'
            ],
            'overview': [
                'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy\'s room.',
                'After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey to bring him home.',
                'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
                'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
                'Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire\'s world-destroying battle station.',
                'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.'
            ],
            'genres': [
                'Animation|Comedy|Family',
                'Animation|Family',
                'Crime|Drama',
                'Crime|Drama',
                'Action|Adventure|Sci-Fi',
                'Adventure|Drama|Sci-Fi'
            ]
        }
        df = pd.DataFrame(data)
        print("✅ Dataset di TEST creato con successo (6 film).")
        return df