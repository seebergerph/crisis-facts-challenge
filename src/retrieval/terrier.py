import typing
import dataclasses
import pandas as pd
import pyterrier as pt
import src.registry as registry


@dataclasses.dataclass
@registry.register("terrier-retriever-config")
class TerrierRetrieverConfig():
    # indexing
    overwrite:bool = True
    index_path:str = ".index"
    index_type:int = 3
    meta:typing.List[str] = None
    meta_lengths:typing.List[int] = None
    properties:typing.Dict[str,str] = None

    # modeling
    wmodel:str = "BM25"
    bo1_expansion:bool = False
    bo1_expansion_terms:int = 3
    bo1_expansion_docs:int = 50


@registry.register("terrier-retriever")
class TerrierRetrieverAdapter():
    def __init__(self, config):
        self.config = registry.create(config)


    def search(self, queries:pd.DataFrame):
        return self.retriever.transform(queries)


    def setup(self, corpus:pd.DataFrame):
        # initialize pyterrier and download jar's
        if not pt.started():
            pt.init()

        # setup document corpus format for indexer
        corpus = corpus.to_dict(orient="records")
        
        # indexing of document corpus
        indexer = pt.IterDictIndexer(
            self.config.index_path, overwrite=self.config.overwrite,
            meta=self.config.meta, meta_lengths=self.config.meta_lengths,
            type=pt.index.IndexingType(self.config.index_type),
            properties=self.config.properties  
        )

        self.indexref = indexer.index(corpus)
        self.index = pt.IndexFactory.of(self.indexref)

        # initialization of retriever
        self.retriever = pt.BatchRetrieve(
            self.index, wmodel=self.config.wmodel, metadata=self.config.meta,
            properties=self.config.properties
        )

        # use out of the box query expansion
        if self.config.bo1_expansion:
            expansion = pt.rewrite.Bo1QueryExpansion(
                self.indexref, fb_terms=self.config.bo1_expansion_terms, 
                fb_docs=self.config.bo1_expansion_docs
            )
            self.retriever = self.retriever >> expansion >>self.retriever
        self.retriever = self.retriever >> pt.pipelines.PerQueryMaxMinScoreTransformer()
        return self


    def clear(self):
        self.index, self.indexref, self.retriever = None, None, None