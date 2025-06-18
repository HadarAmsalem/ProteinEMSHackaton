from Bio.PDB import PDBParser
import numpy as np
import torch
from torch_geometric.data import Data

# Distance threshold for connecting residues
DISTANCE_THRESHOLD = 8.0

# Standard 20 amino acids for one-hot encoding
STANDARD_AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}

def residue_to_one_hot(resname):
    """Convert residue name to one-hot vector of length 20."""
    vec = np.zeros(len(STANDARD_AMINO_ACIDS), dtype=np.float32)
    if resname in AA_TO_INDEX:
        vec[AA_TO_INDEX[resname]] = 1.0
    return vec

def parse_pdb_to_graph(pdb_path):
    """Parse a PDB file and convert it into a graph suitable for GNNs."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    ca_coords = []
    node_features = []

    # Extract CA atoms and build node features
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca = residue['CA']
                    ca_coords.append(ca.coord)
                    one_hot = residue_to_one_hot(residue.get_resname())
                    node_features.append(one_hot)

    ca_coords = np.array(ca_coords)
    node_features = torch.tensor(node_features, dtype=torch.float)
    num_nodes = len(ca_coords)

    # Build edge list and edge attributes (distances)
    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                if dist < DISTANCE_THRESHOLD:
                    edge_index.append([i, j])
                    edge_attr.append([dist])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create PyG Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    return data


