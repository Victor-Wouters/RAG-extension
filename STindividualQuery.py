import numpy as np
#import LMSTUDIO
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import LMSTUDIO
from transformers import pipeline



def process_individual_query(query, model, embeddings, combined_texts):

    threshold = 0.4 # Minimum similarity for a document to be considered relevant

    query_vector = model.encode([query])
    cosine_similarities = cosine_similarity(query_vector, embeddings)
    sorted_indices = np.argsort(cosine_similarities[0])[::-1]
    top_k_indices = [index for index in sorted_indices if cosine_similarities[0][index] > threshold]

    
    if False:
        for rank, index in enumerate(top_k_indices, start=1):
            index = int(index)
            similarity_score = cosine_similarities[0][index]

            if rank <4:
                id = combined_texts[index]['acl_id']   
                print(id)
                print(f"{rank}. SIMILARITY SCORE: {similarity_score:.2f}")

    if not top_k_indices:
        print("No documents found.")
    
    # Use only the single most relevant query
    if True:
        for rank, index in enumerate(top_k_indices, start=1):
            index = int(index)
            similarity_score = cosine_similarities[0][index]
            if rank == 1:
                official_id = combined_texts[index]['acl_id']   
                print(f"{rank}. SIMILARITY SCORE: {similarity_score:.2f}")
                print(f"OFFICAL ID: {official_id}\n")
                
                #summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                # Perform summarization
                #summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
                #print(summary[0]['summary_text'])

                context = " title: " + str(combined_texts[index]['title']) + " author: " + str(combined_texts[index]['author']) + " year: " + str(combined_texts[index]['year']) + " abstract: " + str(combined_texts[index]['abstract']) #+ " full_text: " + str(combined_texts[index]['full_text'])
                print("Context: ")
                print(context)
        final_answer = LMSTUDIO.augment_response(query, context)
        print(final_answer)

    # Use this to provide the 3 top relevant documents as context for the LM
    if False:
        context_parts = []
        for rank, index in enumerate(top_k_indices, start=1):
            if rank < 4:
                doc = merged_dataset[int(index)]
                context_part = f" title: {doc['name']} INGREDIENTS: {doc['ingredients']} STEPS: {doc['steps']} TAGS: {doc['tags']} DESCRIPTION: {doc['description']}"
                context_parts.append(context_part)

        # Join all parts into a single string
        context = " ".join(context_parts)
        print("Context: ")
        print(context)
        final_answer = LMSTUDIO.augment_response(query, context)
        print(final_answer)