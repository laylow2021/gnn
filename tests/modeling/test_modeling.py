import pytest
import torch
from src.gnn.graph.builder import GraphBuilder
from src.gnn.modeling.models import FraudGAT
from src.gnn.modeling.training import train_model
from src.gnn.config.schema import Schema
import pandas as pd
import numpy as np

@pytest.fixture
def simple_graph():
    # Create a minimal graph
    data = {
        Schema.TRANSACTION_ID: ['t1', 't2', 't3'],
        Schema.DATE: pd.to_datetime(['2023-01-01']*3),
        Schema.VOLUME: [100.0, 200.0, 300.0],
        Schema.DIRECTION: ['Inbound', 'Outbound', 'Inbound'],
        Schema.CUSTOMER_NAME: ['Alice', 'Bob', 'Alice'],
        Schema.BANK_NAME: ['BankA', 'BankB', 'BankA'],
        Schema.BANK_ACCOUNT: ['Acc1', 'Acc2', 'Acc1'],
        Schema.RARITY_SCORE: [0.1, 0.2, 0.1],
        Schema.TIMESTAMP_NORM: [0.0, 0.5, 1.0]
    }
    df = pd.DataFrame(data)
    builder = GraphBuilder()
    return builder.build(df)

def test_model_init(simple_graph):
    metadata = simple_graph.metadata()
    model = FraudGAT(hidden_channels=16, out_channels=16, metadata=metadata)
    assert model is not None

def test_forward_pass(simple_graph):
    metadata = simple_graph.metadata()
    model = FraudGAT(hidden_channels=16, out_channels=16, metadata=metadata)
    
    # Initialize lazy layers by passing dummy data or running one forward pass
    # (PyG lazy init happens on first forward)
    output = model(simple_graph.x_dict, simple_graph.edge_index_dict)
    
    assert Schema.NODE_CUSTOMER in output
    assert output[Schema.NODE_CUSTOMER].shape[1] == 16 # Hidden dim

def test_training_loop(simple_graph):
    # Run 1 epoch
    model, z_dict = train_model(simple_graph, hidden_channels=8, epochs=1)
    
    assert model is not None
    assert z_dict is not None
    assert Schema.NODE_CUSTOMER in z_dict
