import numpy as np
#import LMSTUDIO
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import LMSTUDIO
import LMSTUDIOSummarizer
from transformers import pipeline
import spacy
from collections import Counter
from heapq import nlargest


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
    if False:
        for rank, index in enumerate(top_k_indices, start=1):
            index = int(index)
            similarity_score = cosine_similarities[0][index]
            if rank == 1:
                official_id = combined_texts[index]['acl_id']   
                print(f"{rank}. SIMILARITY SCORE: {similarity_score:.2f}")
                print(f"OFFICAL ID: {official_id}\n")
                
                
                
                if False:
                    # Usage in your existing code
                    full_text = str(combined_texts[index]['full_text'])
                    chunks = split_text_into_chunks(full_text, max_chunk_size=4000)  # Adjust max_chunk_size based on your model's limits
                    summaries = [LMSTUDIOSummarizer.summarize_chunk(chunk) for chunk in chunks]
                    combined_summary = ' '.join(summaries)

                    context = " title: " + str(combined_texts[index]['title']) + " author: " + str(combined_texts[index]['author']) + " year: " + str(combined_texts[index]['year']) + " abstract: " + str(combined_texts[index]['abstract']) + 'Summary' + combined_summary #+ " full_text: " + str(combined_texts[index]['full_text'])
                    print("Context: ")
                    print(context)
                elif True:
                    summary = spacy_summarize(str(combined_texts[index]['full_text']), n_sentences=5)
                    context = " title: " + str(combined_texts[index]['title']) + " author: " + str(combined_texts[index]['author']) + " year: " + str(combined_texts[index]['year']) + " abstract: " + str(combined_texts[index]['abstract']) + 'Summary: ' + summary #+ " full_text: " + str(combined_texts[index]['full_text'])
                    print("Context: ")
                    print(context)
                else: 
                    context = " title: " + str(combined_texts[index]['title']) + " author: " + str(combined_texts[index]['author']) + " year: " + str(combined_texts[index]['year']) + " abstract: " + str(combined_texts[index]['abstract']) + " full_text: " + str(combined_texts[index]['full_text'])
                
        final_answer = LMSTUDIO.augment_response(query, context)
        print(final_answer)

    # Use this to provide the 3 top relevant documents as context for the LM
    if True:
        context_parts = []
        for rank, index in enumerate(top_k_indices, start=1):
            if rank < 4:
                doc = combined_texts[int(index)]
                summary = spacy_summarize(str(combined_texts[int(index)]['full_text']), n_sentences=5)
                context_part = f" acl_id: {doc['acl_id']} title: {doc['title']} author: {doc['author']} year: {doc['year']} abstract: {doc['abstract']} Summary: {summary} "
                context_parts.append(context_part)

        # Join all parts into a single string
        context = " ".join(context_parts)
        print("Context: ")
        print(context)
        final_answer = LMSTUDIO.augment_response(query, context)
        print(final_answer)

def split_text_into_chunks(text, max_chunk_size=1000):
    # Split text into chunks without breaking words
    chunks = []
    while text:
        if len(text) > max_chunk_size:
            # Find nearest space to avoid cutting words
            nearest_space = text.rfind(' ', 0, max_chunk_size)
            if nearest_space == -1:
                # No spaces found, forcefully split
                nearest_space = max_chunk_size
            chunks.append(text[:nearest_space])
            text = text[nearest_space:].strip()
        else:
            chunks.append(text)
            break
    return chunks

def spacy_summarize(text, n_sentences=3):

    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    # Count word frequencies (excluding stop words and punctuation)
    word_freq = Counter(token.text.lower() for token in doc if not token.is_stop and not token.is_punct)
    # Score sentences based on word frequencies
    sentence_scores = {sent: sum(word_freq.get(word.text.lower(), 0) for word in sent) for sent in doc.sents}
    # Get the top 'n' sentences
    summary_sentences = nlargest(n_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(sent.text for sent in summary_sentences)
    return summary