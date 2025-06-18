import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from GnnModels import GCN, EGNN  # Make sure both are defined in your GnnModels.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm


def train(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=10, save_path="best_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}", leave=False)
        for data in loop:
            data = data.to(device)
            optimizer.zero_grad()
            # Dynamically handle model signatures
            kwargs = {'x': data.x, 'edge_index': data.edge_index}
            if hasattr(data, 'edge_attr') and model.__class__.__name__ != "GCN":
                kwargs['edge_attr'] = data.edge_attr
            if hasattr(data, 'pos') and model.__class__.__name__ == "EGNN":
                kwargs['pos'] = data.pos
            if hasattr(data, 'batch'):
                kwargs['batch'] = data.batch
            out = model(**kwargs)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
            loop.set_postfix(loss=loss.item()) 

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(save_path))
    return model

def evaluate(model, loader, device, plot_roc=True, roc_output_path="Hackaton/roc_curve.png"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_probs = []  # for ROC
    all_labels = []  # for ROC

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            kwargs = {'x': data.x, 'edge_index': data.edge_index}
            if hasattr(data, 'edge_attr') and model.__class__.__name__ != "GCN":
                kwargs['edge_attr'] = data.edge_attr
            if hasattr(data, 'pos') and model.__class__.__name__ == "EGNN":
                kwargs['pos'] = data.pos
            if hasattr(data, 'batch'):
                kwargs['batch'] = data.batch
            out = model(**kwargs)

            # collect probability for class 1
            prob_class_1 = torch.exp(out[:, 1])  # because output is log_softmax
            all_probs.extend(prob_class_1.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            loss = F.nll_loss(out, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs

    avg_loss = total_loss / total
    acc = correct / total

    if plot_roc:
        plot_roc_curve(all_labels, all_probs, out_file_path=roc_output_path)
        print(f"ROC curve saved to {roc_output_path}")

    return avg_loss, acc

from PDB_to_graph_embbeding import parse_pdb_to_graph
import os
import torch

def load_dataset():
    """
    Loads all PDB files from 'dataset/positives' and 'dataset/negatives' directories.
    Each file is parsed using parse_pdb_to_graph and labeled accordingly.
    Returns a list of PyTorch Geometric Data objects.
    """
    dataset = []
    base_dir = "structures"  # Adjust this path to your dataset directory

    label_map = {
        "af_negatives": 0,
        "af_positives": 1
    }

    for label_name, label_value in label_map.items():
        folder = os.path.join(base_dir, label_name)
        pdb_files = [f for f in os.listdir(folder) if f.endswith(".pdb")]
        for fname in tqdm(pdb_files, desc=f"Loading {label_name}", unit="file"):
            path = os.path.join(folder, fname)
            try:
                graph = parse_pdb_to_graph(path, visualize=False, chain_id='B', save_filtered_pdb=False)
                graph.y = torch.tensor([label_value], dtype=torch.long)
                dataset.append(graph)
            except Exception as e:
                print(f"Skipping {path} due to error: {e}")
    return dataset

def plot_roc_curve(y_test, y_scores, out_file_path="roc_curve.png"):
    """
    Plots a ROC curve of the negative and positive distances
    :param y_test: True data labels (0 for negative, 1 for positive)
    :param y_scores: Distances from the positive training peptides (multiplied by -1)
    :param out_file_path: The path for the output png
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    print(F"AUC: {roc_auc}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out_file_path)


def main(model_type="EGNN"):  # "GCN" or "EGNN"
    dataset = load_dataset()
    torch.manual_seed(42)
    labels = [d.y.item() for d in dataset]
    train_set, temp_set = train_test_split(dataset, test_size=0.3, stratify=labels, random_state=42)
    labels_temp = [d.y.item() for d in temp_set]
    val_set, test_set = train_test_split(temp_set, test_size=0.5, stratify=labels_temp, random_state=42)
    print(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # === Model selection ===
    num_features = train_set[0].x.shape[1]
    edge_feat_dim = train_set[0].edge_attr.shape[1] if hasattr(train_set[0], 'edge_attr') else 0

    if model_type == "GCN":
        model = GCN(num_features=num_features, hidden_dim=64, num_classes=2)
        save_path = "best_gcn_model.pth"
    elif model_type == "EGNN":
        model = EGNN(num_features=num_features, edge_feat_dim=edge_feat_dim, hidden_dim=64)
        save_path = "best_egnn_model.pth"
    else:
        raise ValueError("model_type must be 'GCN' or 'EGNN'")

    # === Training or loading ===
    if os.path.exists(save_path):
        print(f"Loading model from {save_path}...")
        model.load_state_dict(torch.load(save_path))
    else:
        print("Training model from scratch...")
        model = train(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=7, save_path=save_path)
        print(f"Best model saved to {save_path}")

    # === Evaluation ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")

if __name__ == "__main__":
    # Run either model (pass "GCN" or "EGNN")
    main(model_type="EGNN")
    # main(model_type="GCN")
