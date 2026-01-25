import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

# Mock faiss before importing the module under test
sys.modules["faiss"] = MagicMock()

from src.gnn.analysis.clustering import cluster_vectors_with_faiss

class TestClustering(unittest.TestCase):
    def setUp(self):
        # Reset the mock for each test
        sys.modules["faiss"].reset_mock()
        
    def test_cluster_vectors_logic(self):
        """
        Verifies the Connected Components logic given a mocked range_search result.
        """
        # Setup Inputs
        # 4 vectors: A, B, C, D
        # A ~ B (dist > thresh)
        # B ~ C (dist > thresh) -> A, B, C should be one cluster
        # D is isolated
        
        vectors = np.zeros((4, 2)) # Shape doesn't matter for logic test
        ids = ["A", "B", "C", "D"]
        threshold = 0.9
        
        # Mock FAISS Index
        mock_index = MagicMock()
        sys.modules["faiss"].IndexFlatIP.return_value = mock_index
        
        # Mock range_search output: (lims, D, I)
        # lims: [0, 2, 5, 8, 9] (start indices for results)
        # Neighbors (I):
        # A (idx 0): [0, 1] (Self, B) -> count 2
        # B (idx 1): [1, 0, 2] (Self, A, C) -> count 3
        # C (idx 2): [2, 1] (Self, B) -> count 2
        # D (idx 3): [3] (Self) -> count 1
        
        lims = np.array([0, 2, 5, 7, 8])
        # Indices of neighbors flattened
        I = np.array([
            0, 1,    # Neighbors of A
            1, 0, 2, # Neighbors of B
            2, 1,    # Neighbors of C
            3        # Neighbors of D
        ])
        D = np.zeros_like(I, dtype=float) # Distances ignored by our logic (only existence matters)
        
        mock_index.range_search.return_value = (lims, D, I)
        
        # Run function
        clusters = cluster_vectors_with_faiss(vectors, threshold, ids=ids)
        
        # Verify Results
        # Expected: [['A', 'B', 'C'], ['D']] (order within inner lists might vary, outer sorted by size)
        
        self.assertEqual(len(clusters), 2)
        
        # Sort inner lists to compare
        c1 = sorted(clusters[0])
        c2 = sorted(clusters[1])
        
        # Since we sort by size descending
        self.assertEqual(c1, ['A', 'B', 'C'])
        self.assertEqual(c2, ['D'])
        
        # Verify FAISS calls
        sys.modules["faiss"].normalize_L2.assert_called_once()
        sys.modules["faiss"].IndexFlatIP.assert_called_once()
        mock_index.add.assert_called_once()
        mock_index.range_search.assert_called_with(vectors, threshold)

if __name__ == "__main__":
    unittest.main()
