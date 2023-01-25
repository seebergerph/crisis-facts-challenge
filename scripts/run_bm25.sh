#!/bin/bash

python run.py \
--experiment bm25 \
--out_file summary.json \
--config configs/bm25.yaml \
--query_file data/queries.json \
--docs_file data/documents.json \
--overwrite