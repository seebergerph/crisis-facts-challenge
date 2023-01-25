import numpy as np
import pandas as pd
import src.registry as registry


@registry.register("rescorer")
class RescoreModule():
    def __init__(self, ilp=None, mmr=None):
        self.ilp = registry.create(ilp)
        self.mmr = registry.create(mmr)


    def rescore(self, initial_results:pd.DataFrame, history_results:pd.DataFrame=None):
        # calculate scores based on multiple queries
        rescored_results = {"docno": [], "qids": [], "queries": [], "rerank_queries": [], 
                            "text": [], "scores": [], "score": [], "source_type": []}
        for i, group in initial_results.groupby("docno"):
            rescored_results["docno"].append(i)
            rescored_results["qids"].append(group["qid"].tolist())
            rescored_results["queries"].append(group["query"].tolist())
            rescored_results["rerank_queries"].append(group["rerank_query"].tolist())
            rescored_results["text"].append(group["text"].tolist()[0])
            rescored_results["source_type"].append(group["source_type"].tolist()[0])
            rescored_results["scores"].append(group["score"].tolist())
            rescored_results["score"].append(len(group["qid"].tolist()) * np.mean(group["score"].tolist()))
        rescored_results = pd.DataFrame(rescored_results)

        # normalize scores
        min_score = np.min(rescored_results["score"].values)
        max_score = np.max(rescored_results["score"].values)
        rescored_results["score"] = rescored_results["score"].apply(
            lambda score: (score - min_score) / (max_score - min_score)
        )

        # select documents by concepts
        if self.ilp:
            selections = self.ilp.select(rescored_results)
        else:
            selections = rescored_results.copy()

        # rescore documents with reranking
        if self.mmr:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(ngram_range=(1,2))

            embeddings_hist = None
            if history_results is not None:
                texts = pd.concat([selections, history_results], axis=0)["text"].tolist()
                embeddings = vectorizer.fit_transform(texts).toarray()
                embeddings_init = embeddings[:len(selections)]
                embeddings_hist = embeddings[len(selections):]
            else:
                texts = selections["text"].tolist()
                embeddings_init = vectorizer.fit_transform(texts).toarray()

            selections = self.mmr.apply(
                initial_results=selections,
                initial_embs=embeddings_init,
                history_embs=embeddings_hist
            )

        # normalize scores
        min_score = np.min(selections["score"].values)
        max_score = np.max(selections["score"].values)
        selections["score"] = selections["score"].apply(
            lambda score: (score - min_score) / (max_score - min_score)
        )

        # add rank to results
        selections["rank"] = (selections["score"].rank(
            axis=0, method="first", 
            ascending=False
        ) - 1).astype(int)
        return selections