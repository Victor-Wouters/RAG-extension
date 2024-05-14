import datasets
import os
import pickle
from sentence_transformers import SentenceTransformer
import preprocessing
import evaluateIR
import individualQuery

if __name__ == '__main__':

    # Define file paths
    combined_dataset_path = 'combined_dataset.pkl'
    embeddings_path = 'embeddings.pkl'

    anthology_sample = datasets.load_dataset("parquet", data_files="./anthology_sample.parquet")['train']
    model = SentenceTransformer('all-mpnet-base-v2')

    # Check if the combined dataset and embeddings pickle files exist
    if os.path.isfile(combined_dataset_path) and os.path.isfile(embeddings_path):
        # Load combined dataset and embeddings from pickle files
        combined_texts, embeddings = preprocessing.load_in(combined_dataset_path, embeddings_path)
    else:
        combined_texts, embeddings = preprocessing.generate_embedding(combined_dataset_path,embeddings_path,anthology_sample,model)

    #evaluateIR.evaluate_with_Q(model, embeddings, combined_texts, beta = 1)
    
    query = "What is Dynamic Programming Encoding?"
    individualQuery.process_individual_query(query, model, embeddings, combined_texts)