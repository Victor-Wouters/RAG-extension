import datasets
import pickle
from sentence_transformers import SentenceTransformer

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

def load_in(combined_dataset_path, embeddings_path):
    combined_texts = load_pickle(combined_dataset_path)
    embeddings = load_pickle(embeddings_path)
    print("Loaded combined dataset and embeddings from pickle files.")
    return combined_texts, embeddings

def generate_embedding(combined_dataset_path,embeddings_path,anthology_sample,model):
    
    # Apply the merge_columns function to each example in the dataset
    combined_texts = anthology_sample.map(merge_columns)
      
    # Generate embeddings
    embeddings = model.encode(combined_texts['combined'])
    
    # Save the combined dataset and embeddings as pickle files
    save_pickle(combined_texts, combined_dataset_path)
    save_pickle(embeddings, embeddings_path)
    print("Generated and saved combined dataset and embeddings to pickle files.")

    return combined_texts, embeddings

