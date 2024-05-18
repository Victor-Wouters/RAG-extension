from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


import preprocessing

def generate_embedding_SentenceTransformer(embeddings_path,model,combined_texts):
    
    # Generate embeddings
    embeddings = model.encode(combined_texts['combined'])
    
    preprocessing.save_pickle(embeddings, embeddings_path)
    print("Generated and saved combined dataset and embeddings to pickle files.")

    return combined_texts, embeddings


def generate_embedding_WordEmbedding(documents, model):
    doc_embeddings = []

    for doc in documents:
        # Retrieve embeddings for each word in the document if it exists in the model's vocabulary
        embeddings = [model.wv[word] for word in doc if word in model.wv]
        
        if embeddings:  # Check if there are any embeddings
            # keep all word vectors
            doc_embeddings.append(np.array(embeddings))
        else:
            doc_embeddings.append(np.zeros((1,model.vector_size)))
            # Handle the case where none of the words were in the model's vocabulary
            # For example, by appending a zero vector of the same length as the embeddings
    return doc_embeddings  


def compute_bow_wcd(documents, model):
    # Convert documents to a BOW representation
    vectorizer = ListCountVectorizer()
    X = vectorizer.fit_transform(documents)
    bow_counts = X.toarray()
    bow_norms = normalize(bow_counts, norm='l1', axis=1)  # Normalize the BOW counts

    # Compute the weighted average of word vectors for each document
    doc_embeddings = []
    feature_names = vectorizer.get_feature_names_out()
    for doc_idx, doc in enumerate(documents):
        doc_vec = np.zeros(model.vector_size)
        for word_idx, weight in enumerate(bow_norms[doc_idx]):
            word = feature_names[word_idx]
            if word in model.wv:
                doc_vec += model.wv[word] * weight
        doc_embeddings.append(doc_vec)

    return doc_embeddings


class ListCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        # Skip standard text processing because we already did preprocessing, return the token list directly
        return lambda doc: doc