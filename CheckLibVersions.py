from sentence_transformers import SentenceTransformer, __version__ as sentence_transformers_version
from gensim import __version__ as gensim_version
from gensim.models import Word2Vec
import faiss
import heapq
import spacy
import ot

print("Sentence Transformers version:", sentence_transformers_version)
print("Gensim version:", gensim_version)
print("FAISS version:", faiss.__version__)
print("spaCy version:", spacy.__version__)
print("Optimal Transport (POT) version:", ot.__version__)
