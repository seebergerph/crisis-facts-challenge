import os
import yaml
import shutil
import logging
import argparse
import pandas as pd
from collections import Counter

import src.registry as registry
import src.data as data
import src.retrieval as retrieval
import src.reranking as reranking
import src.rescoring as rescoring

from src.logger import logger


def load_config(file):
    config = yaml.load(
        open(file, "r"), 
        Loader=yaml.FullLoader
    )
    return config


def print_data_stats(docs, queries):
    logger.info(f"n_queries: {len(queries)}")
    logger.info(f"n_documents: {len(docs)}")
    for key, group in docs.groupby(by="source_type"):
        logger.info(f"\tn_{key.lower()}: {len(group)}")


def print_result_df_stats(results_df):
    counter = Counter()
    for i, row in results_df.iterrows():
        counter.update(row["qids"])

    logger.info(f"unique documents: {len(results_df)}")
    logger.info(f"unique queries: {len(counter)}")
    for key, count in dict(counter).items():
        logger.info(f"\t{key}: {str(count)}")


def main(args):
    config = load_config(args.config)

    exp_dir = os.path.join("experiments", args.experiment)
    log_dir = os.path.join(exp_dir, "logs")
    out_dir = os.path.join(exp_dir, "results")

    if os.path.isdir(exp_dir) and args.overwrite:
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    stage1_file = os.path.join(out_dir, "retrieval_" + args.out_file)
    stage2_file = os.path.join(out_dir, "reranking_" + args.out_file)
    stage3_file = os.path.join(out_dir, "final_" + args.out_file)

    log_file = os.path.join(log_dir, f"{args.out_file.split('.')[0]}.log")
    logger.addHandler(logging.FileHandler(log_file, "w+"))

    ##### Initialize models

    preprocessor = data.DocumentPreprocessor()
    preprocessor = preprocessor.preprocess

    retriever = registry.create(config["retriever"])
    reranker = registry.create(config["reranker"])
    rescorer = registry.create(config["rescorer"])

    #####

    ##### Stage 0: prepare data

    queries = pd.read_json(args.query_file, lines=True)
    documents = pd.read_json(args.docs_file, lines=True)

    assert "qid" in queries.columns
    assert "query" in queries.columns
    assert "rerank_query" in queries.columns
    assert "docno" in documents.columns
    assert "text" in documents.columns
    assert "source_type" in documents.columns

    documents["text"] = documents.apply(lambda item: preprocessor(
            doc=item["text"], source=item["source_type"]
    ), axis=1)
    print_data_stats(docs=documents, queries=queries)

    # consider past summaries
    if args.history_files is not None and len(args.history_files) > 0:
        history_results  = []
        for file in args.history_files:
            history_docs = pd.read_json(file, lines=True)
            assert "text" in history_docs.columns
            history_results.append(history_docs)
        history_results = pd.concat(history_results, axis=0)
        logger.info(f"n_history_documents: {len(history_results)}")
    else:
        history_results = None

    #####

    ##### Stage 1: retrieval

    logger.info("Stage 1: retrieval")
    retrieval_results = retriever.retrieve(queries, documents)
    retrieval_results = retrieval.drop_exact_duplicates(retrieval_results)
    logger.info(f"\t n_query_doc_pairs: {len(retrieval_results)}")

    #####

    ##### Stage 2: reranking

    logger.info("Stage 2: reranking")
    reranking_results = reranker.rerank(retrieval_results)
    logger.info(f"\t n_query_doc_pairs: {len(reranking_results)}")

    #####

    ##### Stage 3: rescoring / summarization

    logger.info("Stage 3: rescoring & summarization")
    rescoring_results = rescorer.rescore(reranking_results, history_results)
    print_result_df_stats(rescoring_results)

    #####


    ##### Save experiment artifacts

    config_file = os.path.join(exp_dir, "config.yaml")
    with open(config_file, "w") as f: yaml.dump(config, f, indent=2)

    retrieval_results.to_json(stage1_file, orient="records", lines=True)
    reranking_results.to_json(stage2_file, orient="records", lines=True)
    rescoring_results.to_json(stage3_file, orient="records", lines=True)

    #####


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_file", type=str, default="summary.json")
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--docs_file", type=str, required=True)
    parser.add_argument("--history_files", type=str, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    main(arguments)