import os
import pandas as pd


def get_docs_and_queries_local(input_dir, event_id, request):
    date = request["dateString"]
    date = date.replace("-", "_")
    docs_file = os.path.join(input_dir, "documents", event_id, f"{date}.json")
    queries_file = os.path.join(input_dir, "queries", event_id, f"{date}.json")
    documents = pd.read_json(docs_file, lines=True).rename(columns={
        "sourceType": "source_type", 
        "unixTimestamp": "unix_timestamp",
        "streamID": "doc_id"
    })
    queries = pd.read_json(queries_file, lines=True)
    return documents, queries