import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None

def perform_hierarchical_clustering(
    data: pd.DataFrame, 
    features: List[str], 
    n_clusters: Optional[int] = None, 
    **kwargs
) -> pd.Series:
    """
    Performs hierarchical clustering on the provided DataFrame using specified features.
    
    If n_clusters is not provided, it attempts to automatically detect the optimal number 
    of clusters using Silhouette Score (searching between 2 and min(10, n_samples)).
    Alternatively, 'distance_threshold' can be passed in **kwargs to use a distance cut-off.

    Args:
        data: Input pandas DataFrame.
        features: List of column names to use for clustering.
        n_clusters: The number of clusters to find. If None, auto-detection or distance_threshold is used.
        **kwargs: Additional arguments passed to sklearn.cluster.AgglomerativeClustering.

    Returns:
        pd.Series: Cluster labels with the same index as the input DataFrame.
    """
    if data.empty:
        return pd.Series(dtype=int)
        
    X = data[features].values
    
    # Handle small datasets where clustering usually defaults to a single group
    if len(X) < 2:
        return pd.Series([0] * len(X), index=data.index)

    # Automatic detection logic if n_clusters is None
    if n_clusters is None:
        # Check if user provided distance_threshold
        if "distance_threshold" in kwargs:
            model = AgglomerativeClustering(n_clusters=None, **kwargs)
            labels = model.fit_predict(X)
            return pd.Series(labels, index=data.index)
        
        # Otherwise, use Silhouette Score to find best k
        best_k = 2
        best_score = -1
        
        # Search range: 2 to min(10, n_samples)
        max_k = min(10, len(X))
        
        # If we only have 2 samples, we can only do 2 clusters (which is max_k)
        if max_k <= 2:
            best_k = 2
        else:
            for k in range(2, max_k):
                temp_model = AgglomerativeClustering(n_clusters=k, **kwargs)
                temp_labels = temp_model.fit_predict(X)
                
                # Silhouette score requires at least 2 clusters and > 1 sample
                # fit_predict with n_clusters=k guarantees k clusters if n_samples >= k
                if len(np.unique(temp_labels)) > 1:
                    score = silhouette_score(X, temp_labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
        
        n_clusters = best_k

    # Final fit with determined or provided n_clusters
    model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    labels = model.fit_predict(X)
    
    return pd.Series(labels, index=data.index)

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
