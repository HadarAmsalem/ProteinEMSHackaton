import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from GnnModels import GCN, EGNN  # Make sure both are defined in your GnnModels.py
from sklearn.model_selection import train_test_split

def train(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=7, save_path="best_model.pth"):
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
        for data in train_loader:
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

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
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
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def load_dataset():
    """
    Replace with your actual dataset loading.
    Must return a list of Data objects.
    """
    dataset = []  # Your parsed Data objects here
    return dataset

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
