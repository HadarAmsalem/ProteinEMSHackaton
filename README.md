# NES Classifier

This repository contains a trained Graph Neural Network (GNN)-based pipeline for detecting **Nuclear Export Signals (NES)** from protein 3D structures in `.pdb` format.

---

## User Manual

### Input:

- A protein `.pdb` file containing 3D structure, with NES (chain B) and surrounding regions.

### Output:

- Classification result:  
  - `NES POSITIVE` or `NES NEGATIVE`  
- Confidence score (value between 0 and 1)

---

## Installation

Ensure you have the following installed:

- Python ≥ 3.8  
- `torch`, `torch_geometric`, `biopython`, `sklearn`, `matplotlib`, `tqdm`

To install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Place your `.pdb` file under any directory (e.g. `examples/my_protein.pdb`).

2. Run the classifier:

```bash
python user_main.py examples/my_protein.pdb
```

3. Example output:

```
Prediction: NES POSITIVE  
Confidence score: 0.843
```

---

## How It Works

This tool uses a **Graph Neural Network (GNN)** to classify proteins based on the spatial proximity and amino acid types of their residues.

- **Graph Construction**: Each residue becomes a node, and edges are built between nearby residues (within 8Å). Only chain B (NES region) and its surrounding residues (within 15Å) are included in the graph.

- **Node Features**: Each node has a one-hot vector representing the amino acid type and a binary flag indicating if it belongs to the NES chain.

- **Model**: The default model is **EGNN** (Equivariant GNN), trained on labeled positive/negative NES proteins. GCN is also supported.

---

## Model Training

To retrain or experiment with parameters, run:

```bash
python run.py
```

You can modify `run.py` to set:

- `batch_size`  
- `epochs`  
- `learning_rate`  
- `hidden_dim`  
- `dropout`, etc.

Model checkpoints are saved under `Hackaton/`.

---

## Evaluation

After training, the model generates:

- `ROC Curve`: `Hackaton/roc_curve.png`  
- `Boxplot`: `Hackaton/boxplot.png`

These help visualize model separation power between NES-positive and negative samples.

---

## Acknowledgments

This model was developed as part of a protein bioinformatics Hackathon for NES signal detection using 3D structural information and deep learning.
