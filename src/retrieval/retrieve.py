import pandas as pd
import src.registry as registry


@registry.register("retriever")
class RetrieveModule():
    def __init__(self, model, top_k:int=None, threshold:float=None):
        self.model = registry.create(model)
        self.top_k = top_k
        self.threshold = threshold
        

    def _top_k(self, results:pd.DataFrame):
        queries_top_k = []
        for i, group in results.groupby("qid"):
            queries_top_k.append(group.sort_values("rank")[:self.top_k])
        return pd.concat(queries_top_k, axis=0)


    def _threshold(self, results:pd.DataFrame):
        queries_threshold= []
        for i, group in results.groupby("qid"):
            queries_threshold.append(group[group["score"] >= self.threshold])
        return pd.concat(queries_threshold, axis=0)


    def retrieve(self, queries:pd.DataFrame, corpus:pd.DataFrame):
        self.model.setup(corpus)

        # get retrieval results as QxD with score + rank
        retrieval_results = self.model.search(queries)

        if self.threshold:
            retrieval_results = self._threshold(retrieval_results)

        if self.top_k:
            retrieval_results = self._top_k(retrieval_results)

        self.model.clear()
        return retrieval_results