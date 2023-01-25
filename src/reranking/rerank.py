import pandas as pd
import src.registry as registry


@registry.register("reranker")
class RerankModule():
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


    def rerank(self, initial_results:pd.DataFrame):
        self.model.setup()
  
        # get retrieval results as QxD with new score + rank
        rerank_results = self.model.score(initial_results)

        if self.threshold:
            rerank_results = self._threshold(rerank_results)

        if self.top_k:
            rerank_results = self._top_k(rerank_results)

        self.model.clear()
        return rerank_results