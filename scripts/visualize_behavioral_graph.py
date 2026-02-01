import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.gnn.modeling.behavior_gnn import BehaviorGraphBuilder

def generate_mock_data():
    np.random.seed(42)
    n_normal = 50
    n_mule = 5 # Small number for clear visualization
    
    # Normal
    df_normal = pd.DataFrame({
        'cust_id': [f'User_{i}' for i in range(n_normal)],
        'amount': np.random.exponential(1000, n_normal),
        'date': pd.date_range('2024-01-01', periods=n_normal, freq='D'),
        'bank': np.random.choice(['Chase', 'Wells'], n_normal),
        'direction': 'OUT'
    })
    
    # Mules (Synchronized on Jan 5th, Structuring)
    df_mule = pd.DataFrame({
        'cust_id': [f'Mule_{i}' for i in range(n_mule)],
        'amount': [9500] * n_mule, # Exact structuring
        'date': ['2024-01-05'] * n_mule, # Sync Date
        'bank': ['SketchyBank'] * n_mule, # Sync Bank
        'direction': 'OUT'
    })
    
    return pd.concat([df_normal, df_mule]).reset_index(drop=True)

def visualize_suspect(customer_id, data, builder, hops=2, risk_df=None, output_file='ego_graph.png'):
    """
    Visualizes the behavioral ego-graph.
    
    Args:
        risk_df: DataFrame with 'customer_id' and 'is_anomaly' columns.
                 If provided, only 'Anomaly' neighbors are shown in Hop 2.
    """
    print(f"Visualizing {customer_id}...")
    G = nx.Graph()
    
    if customer_id not in builder.cust_map:
        print(f"Customer {customer_id} not found in map.")
        return

    cust_idx = builder.cust_map[customer_id]
    
    # 1. Add Target Customer
    G.add_node(customer_id, color='red', node_type='suspect', size=1000)
    
    # 2. Find Neighbors (Behaviors)
    edge_types = [et for et in data.edge_types if et[0] == 'customer']
    
    for et in edge_types:
        src_type, rel, dst_type = et
        src, dst = data[et].edge_index
        
        # Edges from THIS customer
        my_edges = dst[src == cust_idx].numpy()
        
        # Reverse Map for Label
        if dst_type in builder.node_maps:
            inv_map = {v: k for k, v in builder.node_maps[dst_type].items()}
        else:
            inv_map = {}
            
        for b_idx in my_edges:
            b_label = str(inv_map.get(b_idx, b_idx))
            node_name = f"{dst_type}:{b_label}"
            
            G.add_node(node_name, color='skyblue', node_type='behavior', size=500)
            G.add_edge(customer_id, node_name, relation=rel)
            
            # 3. Hop 2: Find others connected to this behavior
            if hops == 2:
                others = src[dst == b_idx].numpy()
                for o_idx in others:
                    if o_idx == cust_idx: continue
                    
                    # Reverse map customer name
                    inv_cust = {v: k for k, v in builder.cust_map.items()}
                    o_name = inv_cust.get(o_idx, str(o_idx))
                    
                    # --- FILTER LOGIC ---
                    if risk_df is not None:
                        # If risk info provided, check if this neighbor is high risk
                        # Look up
                        if o_name in risk_df['customer_id'].values:
                            is_bad = risk_df.loc[risk_df['customer_id'] == o_name, 'is_anomaly'].iloc[0] == -1
                            if not is_bad:
                                continue # Skip normal people
                        else:
                            # If not in risk df (maybe train/test split?), skip or show grey
                            continue
                    
                    # Color logic: Check if it's a Mule (based on name for demo)
                    color = 'orange'
                    G.add_node(o_name, color=color, node_type='associate', size=300)
                    G.add_edge(node_name, o_name)

    # 4. Plotting
    print(f"Graph constructed. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.3) # k regulates spacing
    
    colors = [nx.get_node_attributes(G, 'color').get(n, 'grey') for n in G.nodes()]
    sizes = [nx.get_node_attributes(G, 'size').get(n, 300) for n in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=sizes, font_size=8, width=0.5)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Target Suspect', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', label='Shared Behavior', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='High Risk Associate', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"High-Risk Behavioral Network of {customer_id}")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # 1. Setup
    df = generate_mock_data()
    print("Mock Data Generated.")
    
    # 2. Build Graph
    config = {
        'customer': 'cust_id',
        'amount': 'amount',
        'date': 'date',
        'bank': 'bank',
        'direction': 'direction',
        'amount_bins': [-1, 9000, 10000, float('inf')] # Catch the 9500 bin
    }
    builder = BehaviorGraphBuilder(config)
    data = builder.build(df)
    
    # Mock Risk DF
    risk_data = pd.DataFrame({
        'customer_id': df['cust_id'].unique(),
        'is_anomaly': [ -1 if 'Mule' in x else 1 for x in df['cust_id'].unique()]
    })
    
    # 3. Visualize Mule_0
    visualize_suspect("Mule_0", data, builder, hops=2, risk_df=risk_data)