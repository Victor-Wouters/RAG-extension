import datasets
import os
import pickle
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import time
import faiss
import numpy as np

import preprocessing
import STevaluateIR
import STindividualQuery
import embed
import WE1individualQuery
import WE1evaluateIR
import WE2individualQuery
import WE2evaluateIR
import ACCevaluateIR
from datasketch import MinHash, MinHashLSH

if __name__ == '__main__':

    anthology_sample = datasets.load_dataset("parquet", data_files="./anthology_sample.parquet")['train']

    combined_dataset_path = 'combined_dataset.pkl'

    if os.path.isfile(combined_dataset_path):
        combined_texts = preprocessing.load_in(combined_dataset_path)
    else:
        # Apply the merge_columns function to each example in the dataset
        combined_texts = anthology_sample.map(preprocessing.merge_columns)
        preprocessing.save_pickle(combined_texts, combined_dataset_path)

    ### SF: Sentence Transformer
    if True:
        model = SentenceTransformer('all-mpnet-base-v2')

        # Define file path
        embeddings_path_SentenceTransformer = 'embeddings_SentenceTransformer.pkl'

        # Check if the combined dataset and embeddings pickle files exist
        if os.path.isfile(embeddings_path_SentenceTransformer):
            # Load combined dataset and embeddings from pickle files
            embeddings = preprocessing.load_in(embeddings_path_SentenceTransformer)
        else:
            embeddings = embed.generate_embedding_SentenceTransformer(embeddings_path_SentenceTransformer,model,combined_texts)

        if False:
            start_time = time.time()
            STevaluateIR.evaluate_with_Q(model, embeddings, combined_texts, beta = 1)
            end_time = time.time()  # Capture end time after function execution
            execution_time = end_time - start_time  # Calculate the execution time

            print(f"The function took {execution_time:.4f} seconds to complete.")

        if True:
            query = "Which versions of the Morfessor tokenizer have been proposed in the literature?"
            STindividualQuery.process_individual_query(query, model, embeddings, combined_texts)

        ## Acceleration
        if False: 
            # Convert embeddings list to a numpy array
            embedding_matrix = np.vstack(embeddings).astype('float32')

            # Build the FAISS index
            dimension = embedding_matrix.shape[1]
            index = faiss.IndexFlatL2(dimension)  # Using L2 distance (euclidean)
            index.add(embedding_matrix)           
            start_time = time.time()
            ACCevaluateIR.evaluate_with_Q(model, embeddings, combined_texts, index, beta = 1)
            end_time = time.time()  # Capture end time after function execution
            execution_time = end_time - start_time  # Calculate the execution time

            print(f"The function took {execution_time:.4f} seconds to complete.")

    ### WE1: Word embedding approach 1: Word Centroid Distance (WCD)
    if False:

        embeddings_path_WordEmbedding = 'embeddings_WordCentroidEmbedding.pkl'

        model_path = "word2vec_model.bin"

        # Check if the model already exists
        if os.path.exists(model_path):
            # Load the trained model from disk
            model = Word2Vec.load(model_path)
            print("Loaded the pre-trained model from disk.")
        else:
            # Prepocess document collection
            documents = preprocessing.tokenize(combined_texts)
            # Train a new model if it doesn't exist
            model = Word2Vec(documents, vector_size=100, window=5, min_count=2, workers=4, sg=1)
            # Save the newly trained model to disk
            model.save(model_path)
            print("Trained and saved a new model.")
        
        if os.path.isfile(embeddings_path_WordEmbedding):
            # Load combined dataset and embeddings from pickle files
            embeddings = preprocessing.load_in(embeddings_path_WordEmbedding)
        else:
            # Prepocess document collection
            documents = preprocessing.tokenize(combined_texts)
            embeddings = embed.compute_bow_wcd(documents,model)
            preprocessing.save_pickle(embeddings, embeddings_path_WordEmbedding)


        if True:
            start_time = time.time()
            WE1evaluateIR.evaluate_with_Q(model, embeddings, combined_texts, beta = 1)
            end_time = time.time()  # Capture end time after function execution
            execution_time = end_time - start_time  # Calculate the execution time

            print(f"The function took {execution_time:.4f} seconds to complete.")

        if False:
            query = "Which versions of the Morfessor tokenizer have been proposed in the literature?"
            query = preprocessing.preprocess_query(query)
            WE1individualQuery.process_individual_query(query, model, embeddings, combined_texts)

    ### WE2: Word embedding approach 2: using Dynamic Time Warping (DTW) 
    if False:
        embeddings_path_WordEmbedding = 'embeddings_WordEmbedding.pkl'

        model_path = "word2vec_model.bin"

        # Check if the model already exists
        if os.path.exists(model_path):
            # Load the trained model from disk
            model = Word2Vec.load(model_path)
            print("Loaded the pre-trained model from disk.")
        else:
            # Prepocess document collection
            documents = preprocessing.tokenize(combined_texts)
            # Train a new model if it doesn't exist
            model = Word2Vec(documents, vector_size=100, window=5, min_count=2, workers=4, sg=1)
            # Save the newly trained model to disk
            model.save(model_path)
            print("Trained and saved a new model.")
        
        if os.path.isfile(embeddings_path_WordEmbedding):
            # Load combined dataset and embeddings from pickle files
            embeddings = preprocessing.load_in(embeddings_path_WordEmbedding)
        else:
             # Prepocess document collection
            documents = preprocessing.tokenize(combined_texts)
            embeddings = embed.generate_embedding_WordEmbedding(documents,model)
            preprocessing.save_pickle(embeddings, embeddings_path_WordEmbedding)

        if False:
            start_time = time.time()
            WE2evaluateIR.evaluate_with_Q(model, embeddings, combined_texts, beta = 1)
            end_time = time.time()  # Capture end time after function execution
            execution_time = end_time - start_time  # Calculate the execution time

            print(f"The function took {execution_time:.4f} seconds to complete.")

        if False:
            query = "Which versions of the Morfessor tokenizer have been proposed in the literature?"
            query = preprocessing.preprocess_query(query)
            WE2individualQuery.process_individual_query(query, model, embeddings, combined_texts)

        