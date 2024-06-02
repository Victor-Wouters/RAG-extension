import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import ot
import embed



def process_individual_query(query, model, embeddings, combined_texts):

    lambda_reg = 0.01  # Regularization parameter for the Sinkhorn algorithm

    query_embeddings = embed.generate_embedding_WordEmbedding([query], model)[0]
    distances = []

    for doc_embedding in embeddings:
        # Calculate the Wasserstein distance with entropy regularization
        M = ot.dist(query_embeddings, doc_embedding, metric='euclidean')
        # Normalize distance matrix
        M /= M.max()
        w1 = np.ones((len(query_embeddings),)) / len(query_embeddings)  # Uniform distribution for query
        w2 = np.ones((len(doc_embedding),)) / len(doc_embedding)  # Uniform distribution for document
        distance = ot.sinkhorn2(w1, w2, M, lambda_reg)
        distances.append(distance)
    
    threshold = 0.68 # Minimum Wasserstein distance for a document to be considered relevant
    sorted_indices = np.argsort(distances)
    top_k_indices = [index for index in sorted_indices if distances[index] < threshold]
    print(top_k_indices)
    for rank, index in enumerate(top_k_indices, start=1):
        index = int(index)
        distance_score = distances[index]
        if rank < 10:
            id = combined_texts[index]['acl_id']
            print(id)
            print(f"{rank}. DISTANCE SCORE: {distance_score:.2f}")

    