import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import ot
import embed



def process_individual_query(query, model, embeddings, combined_texts):

    lambda_reg = 0.01  # Regularization parameter for the Sinkhorn algorithm
    #threshold = 0.01  # Minimum Wasserstein distance for a document to be considered relevant

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
    
    threshold = 0.68  # Documents with a distance under 10% than the lowest distance
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

    # if not top_k_indices:
    #     print("No documents found.")
    
    # # Use only the single most relevant query
    # if True:
    #     for rank, index in enumerate(top_k_indices, start=1):
    #         index = int(index)
    #         similarity_score = cosine_similarities[0][index]
    #         if rank == 1:
    #             official_id = merged_dataset[index]['official_id']   
    #             print(f"{rank}. SIMILARITY SCORE: {similarity_score:.2f}")
    #             print(f"OFFICAL ID: {official_id}\n")
                
    #             context = " NAME: " + str(merged_dataset[index]['name']) + " INGREDIENTS: " + str(merged_dataset[index]['ingredients']) + " STEPS: " + str(merged_dataset[index]['steps']) + " TAGS: " + str(merged_dataset[index]['tags']) + " DESCRIPTION: " + str(merged_dataset[index]['description'])
    #             print("Context: ")
    #             print(context)
    #     final_answer = LMSTUDIO.augment_response(query, context)
    #     print(final_answer)

    # # Use this to provide the 3 top relevant documents as context for the LM
    # if False:
    #     context_parts = []
    #     for rank, index in enumerate(top_k_indices, start=1):
    #         if rank < 4:
    #             doc = merged_dataset[int(index)]
    #             context_part = f" NAME: {doc['name']} INGREDIENTS: {doc['ingredients']} STEPS: {doc['steps']} TAGS: {doc['tags']} DESCRIPTION: {doc['description']}"
    #             context_parts.append(context_part)

    #     # Join all parts into a single string
    #     context = " ".join(context_parts)
    #     print("Context: ")
    #     print(context)
    #     final_answer = LMSTUDIO.augment_response(query, context)
    #     print(final_answer)