#!/bin/bash

python run.py \
--experiment qa \
--out_file summary.json \
--config configs/qa.yaml \
--query_file data/queries.json \
--docs_file data/documents.json \
--overwrite