import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise
import src.registry as registry


def cosine_matrix(docs1, docs2=None):
    if docs2 is not None:
        return pairwise.cosine_similarity(docs1, docs2)
    return pairwise.cosine_similarity(docs1)


def build_sim_matrix(embeddings1, embeddings2=None, similarity='cosine'):
    if similarity == 'cosine':
        return cosine_matrix(embeddings1, embeddings2)
    return np.array([[]])


@registry.register("mmr")
class MMRModule:
    def __init__(self, similarity='cosine', lambda_term=0.8, top_k=None):
        self.similarity = similarity
        self.lambda_term = lambda_term
        self.top_k = top_k


    def _mmr(self, initial_results, sim_matrix, lambda_term, 
             top_k, sim_matrix_history=None):
        if not top_k or top_k > len(initial_results):
            top_k = len(initial_results)

        mmr_scores = []
        selections = []
        candidates = [idx for idx in range(len(initial_results))]
        scores = initial_results["score"].values

        # greedy selection with mmr
        for _ in range(top_k):
            if len(candidates) == 0:
                break

            relevances = scores[candidates]

            redundancies = np.zeros(len(candidates))
            if len(selections) > 0:
                redundancies = sim_matrix[candidates,:][:,selections]
                redundancies = np.max(redundancies, axis=1)

            # if available, take history docs into account
            history = np.zeros(len(redundancies))
            if sim_matrix_history is not None:
                history = sim_matrix_history[candidates,:][:]
                history = np.max(history, axis=1)

            # scoring
            mmr = lambda_term * relevances - \
                  (1 - lambda_term) * np.max(np.vstack([relevances, history]), axis=0)

            score = np.max(mmr)
            idx = candidates[np.argmax(mmr)]

            mmr_scores.append(score)
            selections.append(idx)
            candidates.remove(idx)
        reranked_results = initial_results.iloc[selections, :]
        reranked_results["score"] = np.array(mmr_scores)
        return reranked_results


    def apply(self, initial_results:pd.DataFrame, initial_embs, history_embs=None):
        # create affinity matrix
        sim_matrix = build_sim_matrix(
            embeddings1=initial_embs,
            similarity=self.similarity
        )

        sim_matrix_history = None
        if history_embs is not None:
            sim_matrix_history = build_sim_matrix(
                embeddings1=initial_embs,
                embeddings2=history_embs,
                similarity=self.similarity
            )

        # greedy reranking
        reranked_results = self._mmr(
            initial_results=initial_results,
            sim_matrix=sim_matrix,
            sim_matrix_history=sim_matrix_history,
            lambda_term=self.lambda_term,
            top_k=self.top_k
        )
        return reranked_results