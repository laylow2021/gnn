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

def visualize_suspect(customer_id, data, builder, hops=2, risk_df=None, max_associates=5, output_file='ego_graph.png'):
    """
    Visualizes the behavioral ego-graph.
    
    Args:
        risk_df: DataFrame with 'cust_idx' (or 'customer_id'), 'is_anomaly', and 'risk_score'.
        max_associates: Max number of 2nd-hop neighbors to show per behavior node. 
                        Prevents "hairball" graphs.
    """
    print(f"Visualizing {customer_id}...")
    G = nx.Graph()
    
    if customer_id not in builder.cust_map:
        print(f"Customer {customer_id} not found in map.")
        return

    cust_idx = builder.cust_map[customer_id]
    
    # Pre-compute risk lookup if available
    risk_lookup = {}
    if risk_df is not None:
        # Ensure we can map name -> score or idx -> score
        # The risk_df typically has 'cust_idx' from the pipeline
        if 'cust_idx' in risk_df.columns:
            risk_lookup = dict(zip(risk_df['cust_idx'], risk_df['risk_score']))
        elif 'customer_id' in risk_df.columns:
            # map name to idx first
            name_to_score = dict(zip(risk_df['customer_id'], risk_df['risk_score']))
            risk_lookup = {builder.cust_map[k]: v for k, v in name_to_score.items() if k in builder.cust_map}

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
                
                # --- SORT & LIMIT ---
                # Prioritize showing the HIGHEST RISK associates
                if risk_df is not None:
                    # Sort by risk score (descending)
                    others = sorted(others, key=lambda x: risk_lookup.get(x, -999), reverse=True)
                
                # Limit count
                others = others[:max_associates]
                
                for o_idx in others:
                    if o_idx == cust_idx: continue
                    
                    # Reverse map customer name
                    inv_cust = {v: k for k, v in builder.cust_map.items()}
                    o_name = inv_cust.get(o_idx, str(o_idx))
                    
                    # --- FILTER LOGIC ---
                    if risk_df is not None:
                        # Check if anomaly
                        score = risk_lookup.get(o_idx, -999)
                        # We assume risk_df has 'is_anomaly' or we threshold risk_score?
                        # Let's rely on the previous logic: if passed risk_df, checking is_anomaly column logic 
                        # is tricky if we converted to dict.
                        # Simplest: Just use the fact that we sorted by Risk. 
                        # If the top risk people are not anomalies, then no one is.
                        # But user explicitly wants to hide non-anomalies.
                        
                        # Let's verify 'is_anomaly'
                        is_bad = False
                        if 'is_anomaly' in risk_df.columns and 'cust_idx' in risk_df.columns:
                             # Efficient check?
                             # Better: Pre-compute set of bad indices
                             pass
                        
                        # Optimization:
                        # If we sorted by risk, we are already showing the "worst".
                        # If we want to STRICTLY hide normals:
                        if score < 0: # decision_function < 0 means anomaly usually (if we used raw output), 
                                      # BUT my pipeline flipped it! raw_scores = -iso.decision_function
                                      # So Higher Score = More Anomalous.
                                      # Threshold depends on contamination.
                                      # Instead of guessing threshold, let's use the 'is_anomaly' column if present.
                             pass 
                             
                    # Re-implement strict anomaly filter:
                    if risk_df is not None and 'is_anomaly' in risk_df.columns:
                         # We need to check if o_idx is in the bad set
                         # Let's make a set for O(1) lookup
                         if not hasattr(visualize_suspect, 'bad_indices'):
                             if 'cust_idx' in risk_df.columns:
                                 visualize_suspect.bad_indices = set(risk_df[risk_df['is_anomaly'] == -1]['cust_idx'].values)
                             else:
                                 # map names
                                 bad_names = set(risk_df[risk_df['is_anomaly'] == -1]['customer_id'].values)
                                 visualize_suspect.bad_indices = {builder.cust_map[n] for n in bad_names if n in builder.cust_map}
                        
                         if o_idx not in visualize_suspect.bad_indices:
                             continue

                    
                    # Color logic
                    color = 'orange'
                    G.add_node(o_name, color=color, node_type='associate', size=300)
                    G.add_edge(node_name, o_name)

    # Clean up static attribute if used (hacky but works for script)
    if hasattr(visualize_suspect, 'bad_indices'):
        del visualize_suspect.bad_indices

    # 4. Plotting
    print(f"Graph constructed. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.3) 
    
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
        'is_anomaly': [ -1 if 'Mule' in x else 1 for x in df['cust_id'].unique()],
        'risk_score': np.random.rand(len(df['cust_id'].unique())) # Mock scores
    })
    # Add cust_idx
    risk_data['cust_idx'] = risk_data['customer_id'].map(builder.cust_map)
    
    # 3. Visualize Mule_0
    visualize_suspect("Mule_0", data, builder, hops=2, risk_df=risk_data, max_associates=3)
