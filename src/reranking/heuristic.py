import re
import stanza
import dataclasses
import numpy as np
import pandas as pd
import pyterrier as pt
import src.registry as registry
import src.data.patterns as patterns
import src.data.preprocess as preprocess
from collections import defaultdict

stanza.download("en")


@dataclasses.dataclass
@registry.register("heuristic-reranker-config")
class HeuristicRerankerConfig():
    # modeling
    use_gpu:bool = False
    matching:bool = True
    batch_size:int = 64


@registry.register("heuristic-reranker")
class HeuristicReranker():
    def __init__(self, config):
        self.config = registry.create(config)
        self.pipeline = stanza.Pipeline(
            lang="en",
            processors="tokenize,ner",
            use_gpu=self.config.use_gpu 
        )


    def score(self, initial_results:pd.DataFrame):
        fn = self._boe_score
        rerank = pt.apply.doc_score(fn, batch_size=self.config.batch_size) \
            >> pt.pipelines.PerQueryMaxMinScoreTransformer()
        rerank_results = rerank.transform(initial_results)
        return rerank_results


    def setup(self):
        pass


    def clear(self):
        pass


    def _boe_score(self, initial_results:pd.DataFrame):
        score_fn = self._match_score if self.config.matching else self._freq_score
        scores = []
        for i, row in initial_results.iterrows():
            qid = row["qid"]
            did = row["docno"]
            doc = row["text"]

            keywords = patterns.QUERY_PATTERNS[qid]["keywords"]
            entities = patterns.QUERY_PATTERNS[qid]["entities"]

            entities_match = self._entities(doc, did)
            keywords_match = self._keywords(doc, keywords)

            entities_match = {
                entity: count 
                for entity,count in entities_match.items() 
                if entity in entities
            }

            scores.append(score_fn(entities_match, keywords_match))
        return np.array(scores)


    def _preprocess(self, doc):
        doc = preprocess.drop_punctuation(doc)
        tokens = preprocess.drop_stop_words(doc.split())
        tokens = preprocess.stemming(tokens)
        return " ".join(tokens)


    def _entities(self, doc, doc_id):
        matches = defaultdict(int)
        for entity in self.pipeline(doc).ents:
            matches[entity.type] += 1
        return matches


    def _keywords(self, doc, keywords):
        doc = self._preprocess(doc)
        keywords = self._preprocess(keywords)
        keywords = list(set(keywords.split()))

        pattern = re.compile(r"\b(" + (r"|".join(keywords)) + r")\b")
        matches = defaultdict(int)
        for match in pattern.findall(doc):
            matches[match] += 1
        return matches


    def _match_score(self, entities, keywords):
        score = 0.
        score += len(entities)
        score += len(keywords)
        return score


    def _freq_score(self, entities, keywords):
        score = 0.
        score += np.sum([count for count in entities.values()])
        score += np.sum([count for count in keywords.values()])
        return score