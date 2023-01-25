import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def drop_exact_duplicates(results:pd.DataFrame):
    results_filtered = []
    for i, group in results.groupby("qid"):
        group = group.drop_duplicates(
            subset=["text"], keep="first"
        )
        results_filtered.append(group)
    results_filtered = pd.concat(results_filtered, axis=0)
    return results_filtered


def drop_near_duplicates(results:pd.DataFrame, threshold:float=0.95, ngrams=(1,2)):
    results_filtered = []

    for i, group in results.groupby("qid"):
        # extract ngram vectors
        texts = group["text"]
        vectorizer = CountVectorizer(ngram_range=ngrams)
        ngram_features = vectorizer.fit_transform(texts).toarray()
        
        # calculate triangular similarity matrix
        sim_matrix = cosine_similarity(ngram_features, ngram_features)
        np.fill_diagonal(sim_matrix, 0)
        sim_matrix = np.triu(sim_matrix)
        sim_matrix = pd.DataFrame(sim_matrix)
        
        # always keep first row of near duplicates
        indices = sim_matrix.index.values
        for i, row in sim_matrix.iterrows():
            mask = row[indices] < threshold
            indices = indices[mask]
        group = group.iloc[indices, :]
    results_filtered = pd.concat(results_filtered, axis=0)
    return results_filtered