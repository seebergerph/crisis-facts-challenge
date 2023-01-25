import dataclasses
import numpy as np
import pandas as pd
import pyterrier as pt
import src.registry as registry
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclasses.dataclass
@registry.register("terrier-reranker-config")
class TerrierRerankerConfig():
    # modeling
    model_name_or_path:str = "paraphrase-MiniLM-L6-v2"
    cross_encoder:bool = False
    max_length:int = 128
    batch_size:int = 64
    device:str = None
    query_col:str = "query"


@registry.register("terrier-reranker")
class TerrierRerankerAdapter():
    def __init__(self, config):
        self.config = registry.create(config)
        if self.config.cross_encoder:
            self.model = CrossEncoder(
                self.config.model_name_or_path,
                max_length=self.config.max_length,
                device=self.config.device
            )
        else:
            self.model = SentenceTransformer(
                self.config.model_name_or_path,
                device=self.config.device
            )


    def score(self, initial_results:pd.DataFrame):
        if self.config.cross_encoder:
            fn = self._cross_encoder
        else:
            fn = self._bi_encoder
        rerank = pt.apply.doc_score(fn, batch_size=self.config.batch_size) \
            >> pt.pipelines.PerQueryMaxMinScoreTransformer()
        rerank_results = rerank.transform(initial_results)
        return rerank_results


    def setup(self):
        # initialize pyterrier and download jar's
        if not pt.started():
            pt.init()
            

    def clear(self):
        pass


    def _cross_encoder(self, initial_results:pd.DataFrame):
        scores = self.model.predict(list(zip(
            initial_results[self.config.query_col].values,
            initial_results["text"].values
        )))
        return scores


    def _bi_encoder(self, initial_results:pd.DataFrame):
        from sentence_transformers.util import cos_sim
        query_embs = self.model.encode(initial_results[self.config.query_col].values)
        doc_embs = self.model.encode(initial_results["text"].values)
        scores = cos_sim(query_embs, doc_embs)
        return scores[0]