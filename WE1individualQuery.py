import numpy as np
#import LMSTUDIO
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import embed



def process_individual_query(query, model, embeddings, combined_texts):

    threshold = 0.1 # Minimum similarity for a document to be considered relevant

    query_vector = embed.compute_bow_wcd([query], model)
    cosine_similarities = cosine_similarity(query_vector, embeddings)
    sorted_indices = np.argsort(cosine_similarities[0])[::-1]
    top_k_indices = [index for index in sorted_indices if cosine_similarities[0][index] > threshold]

    # cosine_similarities = util.dot_score(query_vector, embeddings)[0]
    # print("Similarity:", util.dot_score(query_vector, embeddings))
    # cosine_similarities_np = cosine_similarities.cpu().numpy()
    # sorted_indices = np.argsort(cosine_similarities_np)[::-1]
    # top_k_indices = [index for index in sorted_indices if cosine_similarities[index] > threshold]
    
    if True:
        for rank, index in enumerate(top_k_indices, start=1):
            index = int(index)
            similarity_score = cosine_similarities[0][index]
            #similarity_score = cosine_similarities[index].item()
            if rank <4:
                id = combined_texts[index]['acl_id']   
                print(id)
                print(f"{rank}. SIMILARITY SCORE: {similarity_score:.2f}")

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