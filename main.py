import sys
# Importiamo le nostre classi personalizzate
from data_loader import DataLoader
from embeddings import EmbeddingGenerator
from recommender import MovieRecommender

def main():
    print("============================================")
    print("🎬  MOVIE PLOT RECOMMENDER SYSTEM (BERT)  🎬")
    print("============================================")

    # 1. Caricamento Dati
    # Se avete il file vero, cambiate 'movies.csv' con il percorso giusto
    loader = DataLoader(filepath='movies_metadata.csv') 
    df = loader.load_data()

    # Per velocizzare i test, se il dataset è enorme, prendiamo solo i primi 1000
    if len(df) > 5000:
        print("⚠️ Dataset grande: taglio ai primi 1000 film per velocità.")
        df = df.head(1000)

    # 2. Generazione Embeddings (Usiamo BERT come richiesto dai prof)
    # Cambia method='tfidf' se vuoi provare la baseline
    embedder = EmbeddingGenerator(method='bert') 
    embeddings = embedder.fit_transform(df['overview'].tolist())

    # 3. Inizializzazione Motore Raccomandazione
    recsys = MovieRecommender(df, embeddings)
    recsys.compute_similarity()

    # 4. Loop Interattivo
    while True:
        print("\n" + "-"*40)
        query = input("Inserisci il titolo di un film (o 'q' per uscire): ").strip()
        
        if query.lower() == 'q':
            print("Chiusura sistema. Ciao!")
            break
            
        recommendations, msg = recsys.recommend(query, top_n=3)
        
        if recommendations is not None:
            print(f"\n🌟 Film simili a '{query}':")
            # Stampa carina del dataframe senza indice
            print(recommendations.to_string(index=False))
        else:
            print(f"❌ Errore: {msg}")
            # Suggerimento: mostra titoli disponibili se è il dataset dummy
            if len(df) < 20:
                print(f"Titoli disponibili: {', '.join(df['title'].tolist())}")

if __name__ == "__main__":
    main()