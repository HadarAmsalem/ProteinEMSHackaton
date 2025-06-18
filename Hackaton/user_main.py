import torch
import sys
import os
from PDB_to_graph_embbeding import parse_pdb_to_graph
from GnnModels import EGNN  # or GCN if preferred
import torch.nn.functional as F

def classify_pdb(pdb_path, model_path="best_egnn_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load and parse the PDB file into a graph
    try:
        data = parse_pdb_to_graph(pdb_path, visualize=False, save_filtered_pdb=False)
    except Exception as e:
        print(f"Failed to parse PDB file: {e}")
        return

    data = data.to(device)
    num_features = data.x.shape[1]
    edge_feat_dim = data.edge_attr.shape[1] if hasattr(data, 'edge_attr') else 0

    # Step 2: Load the trained model
    model = EGNN(num_features=num_features, edge_feat_dim=edge_feat_dim, hidden_dim=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    # Step 3: Predict
    with torch.no_grad():
        kwargs = {'x': data.x, 'edge_index': data.edge_index, 'batch': data.batch}
        if hasattr(data, 'edge_attr'):
            kwargs['edge_attr'] = data.edge_attr
        if hasattr(data, 'pos'):
            kwargs['pos'] = data.pos
        out = model(**kwargs)
        probs = torch.exp(out)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][1].item()

    # Step 4: Output results
    label_str = "NES POSITIVE" if pred == 1 else "NES NEGATIVE"
    print(f"Prediction: {label_str}")
    print(f"Confidence score: {confidence:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_model_on_pdb.py <path_to_pdb_file>")
        sys.exit(1)
    pdb_file = sys.argv[1]
    print(f"Classifying PDB file: {pdb_file}")
    classify_pdb(pdb_file)
