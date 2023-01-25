import stanza
import numpy as np
import src.registry as registry
import src.data.preprocess as preprocess
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


stanza.download("en")

DEFAULT_NER_TAGS = [
    "CARDINAL", "DATE", "EVENT", "FAC", "GPE", 
    "LOC", "MONEY", "ORDINAL", "ORG", "PERCENT", 
    "PRODUCT", "QUANTITY", "TIME"
]


@registry.register("ngram-concepts")
class NgramConceptsCreator():
    def __init__(self, ngram_range=(2,2), stopwords="english", 
                 lowercase=True, min_count=1):
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=stopwords,
            lowercase=lowercase,
            binary=True
        )
        self.min_count = min_count


    def extract(self, docs):
        feats = self.vectorizer.fit_transform(docs)
        feats = np.asarray(feats.todense())

        ngrams = np.array(self.vectorizer.get_feature_names_out())
        counts = np.sum(feats, axis=0)

        min_idx = np.argwhere(counts >= self.min_count)
        min_idx = min_idx.reshape(-1)
        
        feats = feats[:,min_idx]
        ngrams = ngrams[min_idx]
        counts = counts[min_idx]

        weights = {
            ngrams[i]: num
            for i, num in enumerate(counts)
        }

        concepts = []
        mask = feats >= 1
        for i in range(len(mask)):
            feats_mask = mask[i]
            concepts.append(ngrams[feats_mask])
        return concepts, weights


@registry.register("entity-concepts")
class EntityConceptsCreator():
    def __init__(self, ner_tags=DEFAULT_NER_TAGS, use_gpu=False):
        self.pipeline = stanza.Pipeline(
            lang="en",
            processors="tokenize,ner",
            use_gpu=use_gpu 
        )

        self.possible_ner_tags = self._get_ner_tags()

        if ner_tags is not None:
            self.possible_ner_tags = [
                ner_tag
                for ner_tag in self.possible_ner_tags
                if ner_tag in ner_tags
            ]


    def _get_ner_tags(self):
        processor = self.pipeline.processors["ner"]
        ner_tags = processor.get_known_tags()
        return ner_tags


    def _extract(self, doc):
        def process(entity):
            tokens = entity.lower().split()
            tokens = preprocess.drop_stop_words(tokens)
            return " ".join(tokens)

        entities = self.pipeline(doc).ents
        concepts = [
            process(entity.text)
            for entity in entities
            if entity.type in self.possible_ner_tags
        ]
        return concepts


    def extract(self, docs):
        concepts = [
            self._extract(doc)
            for doc in docs
        ]

        unique_per_doc = [list(set(cs)) for cs in concepts]
        unique_per_doc = [item for items in unique_per_doc for item in items]
        counter = Counter(unique_per_doc)

        weights = {
            concept: num
            for concept, num in dict(counter).items()
        }
        return concepts, weights