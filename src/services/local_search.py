class LocalSearchService:
    def __init__(self, df):
        self.df = df

    def search(self, query):
        # Ricerca parziale nel dataframe
        return self.df[self.df['title'].str.contains(query, case=False, regex=False)]