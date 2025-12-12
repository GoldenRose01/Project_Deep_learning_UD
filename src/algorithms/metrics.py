def calculate_genre_overlap(df, recommender, n_samples=50):
    samples = df.sample(min(n_samples, len(df)))
    matches = 0
    total = 0

    for _, row in samples.iterrows():
        recs = recommender.recommend_single(row['title'])
        if recs is None: continue

        src_genres = set(str(row['genres']).split('|'))
        for _, rec in recs.iterrows():
            rec_genres = set(str(rec['genres']).split('|'))
            if not src_genres.isdisjoint(rec_genres):
                matches += 1
            total += 1

    return matches / total if total > 0 else 0