import numpy as np
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None

def cluster_vectors_with_faiss(
    vectors: np.ndarray, 
    threshold: float, 
    ids: List[str] = None
) -> List[List[str]]:
    """
    Clusters vectors into groups based on a cosine similarity threshold using FAISS.
    Uses a Connected Components approach (Single Linkage): if A is similar to B, 
    and B is similar to C, they form a cluster {A, B, C}.

    Args:
        vectors: Numpy array of shape (n_samples, n_features).
        threshold: Cosine similarity threshold (0.0 to 1.0). 
                   Pairs with similarity >= threshold are connected.
        ids: Optional list of IDs corresponding to the vectors. 
             If None, returns lists of integer indices.

    Returns:
        List of clusters, where each cluster is a list of IDs (or indices).
    """
    if faiss is None:
        raise ImportError(
            "The 'faiss' library is required for this function. "
            "Please install it using `pip install faiss-cpu` or `pip install faiss-gpu`."
        )

    if vectors.ndim != 2:
        raise ValueError("Input vectors must be 2-dimensional.")

    n_samples, d = vectors.shape
    
    # 1. Normalize vectors for Cosine Similarity
    # FAISS IndexFlatIP calculates inner product. 
    # Normalized Dot Product == Cosine Similarity.
    faiss.normalize_L2(vectors)

    # 2. Build Index
    index = faiss.IndexFlatIP(d)
    index.add(vectors)

    # 3. Range Search
    # Returns:
    # lims: starting index in D and I for each query
    # D: distances (similarities)
    # I: indices of neighbors
    lims, D, I = index.range_search(vectors, threshold)

    # 4. Build Graph / Connected Components
    # We use a Union-Find (Disjoint Set) data structure for efficiency
    parent = list(range(n_samples))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Iterate through search results
    for i in range(n_samples):
        start = lims[i]
        end = lims[i+1]
        neighbors = I[start:end]
        
        for neighbor_idx in neighbors:
            # range_search includes the node itself (sim=1.0), which is harmless for Union-Find
            # but we can skip if i == neighbor_idx to save a tiny bit of time
            if i < neighbor_idx: # processing pairs once is enough
                union(i, neighbor_idx)

    # 5. Group by Root
    clusters_dict: Dict[int, List[Union[int, str]]] = {}
    
    for i in range(n_samples):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        
        val = ids[i] if ids is not None else i
        clusters_dict[root].append(val)

    # Convert to list of lists
    clusters = list(clusters_dict.values())
    
    # Sort clusters by size (descending) for convenience
    clusters.sort(key=len, reverse=True)

    return clusters
