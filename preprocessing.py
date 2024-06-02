import datasets
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import ot

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to combine all columns in an additional column "combined"
def merge_columns(dataset):
    combined_text = ' '.join([str(dataset[key]) for key in dataset])
    return {'combined': combined_text}

# Function to save data to a pickle file
def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# Function to load data from a pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_in(pickle_file):
    in_memory = load_pickle(pickle_file)
    print("Loaded in pickle file.")
    return in_memory

def tokenize(combined_texts):

    stop_words = set(stopwords.words('english'))

    # Tokenize and remove stop words
    documents = [
    [word for word in word_tokenize(doc['combined'].lower()) if word not in stop_words and word.isalpha()]
    for doc in combined_texts]

    return documents

def preprocess_query(query):
    
    stop_words = set(stopwords.words('english'))
    
    # Tokenize and remove stop words from the query
    query_tokens = [word for word in word_tokenize(query.lower()) if word not in stop_words and word.isalpha()]

    return query_tokens

def compute_wasserstein_distance(query_embedding, doc_embedding, lambda_reg=0.1):
    M = ot.dist(query_embedding, doc_embedding, metric='euclidean')
    M /= M.max()
    w1 = np.ones((len(query_embedding),)) / len(query_embedding)  # Uniform distribution for query
    w2 = np.ones((len(doc_embedding),)) / len(doc_embedding)  # Uniform distribution for document
    distance = ot.sinkhorn2(w1, w2, M, lambda_reg)
    return distance


