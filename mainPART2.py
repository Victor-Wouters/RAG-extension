import datasets


anthology_sample = datasets.load_dataset("parquet", data_files="./anthology_sample.parquet")['train']

print(anthology_sample[0]['full_text'])