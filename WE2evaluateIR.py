import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ot
import embed


def evaluate_with_Q(model, embeddings, combined_texts, beta = 1):
    with open('acl_anthology_queries.json', 'r') as file:
        data = json.load(file)

    # MACRO AVERAGES
    macro_precisions = []
    macro_recalls = []
    macro_F1s = []
    average_precisions = []
    
    # MICRO AVERAGES
    TP = 0
    FP = 0
    FN = 0

    for query in data['queries']:
        q = query['q']
        r_benchmark_set = set(query['r'])
        
        lambda_reg = 0.1
        distances = []

        query_embeddings =  embed.generate_embedding_WordEmbedding([q],model)[0]
        for doc_embedding in embeddings:
            # Calculate the Wasserstein distance with entropy regularization
            M = ot.dist(query_embeddings, doc_embedding, metric='euclidean')
            # Normalize distance matrix
            M /= M.max()
            w1 = np.ones((len(query_embeddings),)) / len(query_embeddings)  # Uniform distribution for query
            w2 = np.ones((len(doc_embedding),)) / len(doc_embedding)  # Uniform distribution for document
            distance = ot.sinkhorn2(w1, w2, M, lambda_reg)
            distances.append(distance)
        
        threshold = 0.65  # Documents with a distance under 10% than the lowest distance
        sorted_indices = np.argsort(distances)
        top_k_indices = [index for index in sorted_indices if distances[index] < threshold]
        r = [combined_texts[int(index)]['acl_id'] for index in top_k_indices]
        r_set = set(r)
        
        
        # Calculate and store precision and recall
        precision = calculate_precision(r_set, r_benchmark_set, len(r))
        recall = calculate_recall(r_set, r_benchmark_set, len(query['r']))
        if precision + recall > 0:
            F1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        else:
            F1 = 0

        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_F1s.append(F1)


        # Update counts for TP, FP, and FN for micro average metrics
        TP += len(r_set & r_benchmark_set)
        FP += len(r_set - r_benchmark_set)
        FN += len(r_benchmark_set - r_set)

        # Calculate Average Precision (AP) for MAP
        ap = calculate_average_precision(r, query['r'])
        average_precisions.append(ap)

    # Calculate and print macro-averages
    macro_average_precision = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0
    macro_average_recall = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0
    macro_average_F1 = sum(macro_F1s) / len(macro_F1s) if macro_F1s else 0

    print(f"Macro-average precision: {macro_average_precision}")
    print(f"Macro-average recall: {macro_average_recall}")
    print(f"Macro-average F1: {macro_average_F1}")


    # Calculate micro-averaged precision and recall
    micro_precision = TP / (TP + FP) if TP + FP > 0 else 0
    micro_recall = TP / (TP + FN) if TP + FN > 0 else 0

    # Calculate micro-averaged F1 score
    if micro_precision + micro_recall > 0:
        micro_F1 = (1 + beta**2) * (micro_precision * micro_recall) / ((beta**2 * micro_precision) + micro_recall)
    else:
        micro_F1 = 0

    print(f"Micro-average precision: {micro_precision}")
    print(f"Micro-average recall: {micro_recall}")
    print(f"Micro-average F1: {micro_F1}")

    # Calculate Mean Average Precision (MAP)
    MAP = sum(average_precisions) / len(average_precisions) if average_precisions else 0

    print(f"Mean Average Precision (MAP): {MAP}")



def calculate_precision(retrieved_set, benchmark_set, retrieved_len):
    if retrieved_len > 0:
        return len(retrieved_set & benchmark_set) / retrieved_len
    return 0

def calculate_recall(retrieved_set, benchmark_set, benchmark_len):
    if benchmark_len > 0:
        return len(retrieved_set & benchmark_set) / benchmark_len
    return 0

def calculate_average_precision(retrieved, relevant): # Calculates the average precision for a single query.

    relevant_hits = 0
    cumulative_precision = 0
    relevant_set = set(relevant)

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            relevant_hits += 1
            precision_at_i = relevant_hits / (i + 1)
            cumulative_precision += precision_at_i


    return cumulative_precision / len(relevant_set) if relevant_hits > 0 else 0

if __name__ == '__main__':
    with open('acl_anthology_queries.json', 'r') as file:
        data = json.load(file)
    for query in data['queries']:
        q = query['q']
        r_benchmark_set = set(query['r'])
        print(q)
        print(r_benchmark_set)