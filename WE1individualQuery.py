import numpy as np
#import LMSTUDIO
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import embed



def process_individual_query(query, model, embeddings, combined_texts):

    threshold = 0.55 # Minimum similarity for a document to be considered relevant

    query_vector = embed.compute_bow_wcd([query], model)
    cosine_similarities = cosine_similarity(query_vector, embeddings)
    sorted_indices = np.argsort(cosine_similarities[0])[::-1]
    top_k_indices = [index for index in sorted_indices if cosine_similarities[0][index] > threshold]

    
    if True:
        for rank, index in enumerate(top_k_indices, start=1):
            index = int(index)
            similarity_score = cosine_similarities[0][index]
            
            if rank <4:
                id = combined_texts[index]['acl_id']   
                print(id)
                print(f"{rank}. SIMILARITY SCORE: {similarity_score:.2f}")

