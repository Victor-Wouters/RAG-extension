import json
import datasets

R_NUMBER_SEED = 810913 # Replace this with your own student number: 0810913
DOCS_TO_ADD = 1000
query_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_queries.parquet")["train"]
all_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_full.parquet")["train"]

random_documents = all_documents.shuffle(seed=R_NUMBER_SEED).take(DOCS_TO_ADD)

anthology_sample = datasets.concatenate_datasets([query_documents, random_documents]).shuffle(seed=R_NUMBER_SEED)

anthology_sample.to_parquet("./anthology_sample.parquet")

anthology_sample = datasets.load_dataset("parquet", data_files="./anthology_sample.parquet")['train']

print(anthology_sample[0])

queries = json.load(open("./acl_anthology_queries.json", "r"))
for k, v in queries["queries"][0].items():
  print(f"{k.upper()}: {v}\n")