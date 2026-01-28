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

from src.gnn.analysis.clustering import perform_hierarchical_clustering
import pandas as pd

class TestHierarchicalClustering(unittest.TestCase):
    def test_basic_clustering(self):
        """Test with explicit n_clusters."""
        df = pd.DataFrame({
            'f1': [1, 1, 10, 10],
            'f2': [1, 1, 10, 10]
        })
        features = ['f1', 'f2']
        
        # Expect 2 clusters: {0,1} and {2,3}
        labels = perform_hierarchical_clustering(df, features, n_clusters=2)
        
        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_auto_detection(self):
        """Test automatic cluster detection."""
        # Create 3 distinct groups
        df = pd.DataFrame({
            'A': [0, 0, 10, 10, 20, 20],
            'B': [0, 0, 10, 10, 20, 20]
        })
        features = ['A', 'B']
        
        # Auto detection should likely find 3 clusters
        labels = perform_hierarchical_clustering(df, features)
        
        unique_labels = labels.unique()
        self.assertTrue(len(unique_labels) >= 2)
        # Ideally it finds 3, but silhouette can be tricky. 
        # With perfectly separated data, it should be robust.
        self.assertEqual(len(unique_labels), 3)

    def test_groupby_integration(self):
        """Test usage within a groupby apply."""
        df = pd.DataFrame({
            'group': ['g1', 'g1', 'g1', 'g1', 'g2', 'g2'],
            'val':   [1, 2, 100, 101, 5, 6]
        })
        
        # Function to apply
        def cluster_group(x):
            return perform_hierarchical_clustering(x, features=['val'], n_clusters=2)
            
        result = df.groupby('group').apply(cluster_group)
        
        # Result should be a Series with MultiIndex (group, original_index) or similar, 
        # depending on pandas version. But let's check values.
        # For g1: (1,2) should be together, (100,101) together.
        # For g2: (5,6) are only 2 points, so if n_clusters=2, they might be split 
        # or if logic handles < 2 points (here we have 2, so 2 clusters = each its own).
        
        # Check g1
        g1_labels = result.loc['g1']
        # The indices in g1 are 0,1,2,3
        # 1 and 2 (idx 0,1) should be same cluster
        self.assertEqual(g1_labels.loc[0], g1_labels.loc[1])
        # 100 and 101 (idx 2,3) should be same cluster
        self.assertEqual(g1_labels.loc[2], g1_labels.loc[3])
        # groups should differ
        self.assertNotEqual(g1_labels.loc[0], g1_labels.loc[2])

    def test_small_data(self):
        """Test handling of 0 or 1 rows."""
        df_empty = pd.DataFrame({'a': []})
        res_empty = perform_hierarchical_clustering(df_empty, ['a'])
        self.assertTrue(res_empty.empty)
        
        df_single = pd.DataFrame({'a': [1]})
        res_single = perform_hierarchical_clustering(df_single, ['a'])
        self.assertEqual(len(res_single), 1)
        self.assertEqual(res_single.iloc[0], 0)

if __name__ == "__main__":
    unittest.main()
