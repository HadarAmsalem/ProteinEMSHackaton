from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import os

# Distance threshold (Å) for edge connection
DISTANCE_THRESHOLD = 8.0

# Distance threshold for proximity to NES chain
NES_PROXIMITY_THRESHOLD = 15.0

# Amino acid one-hot encoding
STANDARD_AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}

def residue_to_one_hot(resname):
    """
    Convert a residue name (3-letter amino acid code) to a one-hot encoded vector.
    The result is a vector of length 20 with a 1 in the position corresponding to the amino acid type.
    """
    vec = np.zeros(len(STANDARD_AMINO_ACIDS), dtype=np.float32)
    if resname in AA_TO_INDEX:
        vec[AA_TO_INDEX[resname]] = 1.0
    return vec

def get_ca_coord(residue):
    """
    Return the 3D coordinates of the C-alpha (CA) atom of a residue.
    If the CA atom is missing, return None.
    """
    return residue['CA'].coord if 'CA' in residue else None

def filter_structure_by_chain_and_proximity(structure, chain_id='B', distance_threshold=NES_PROXIMITY_THRESHOLD):
    """
    Extract residues from a structure that are either:
    - part of the specified chain (default: chain 'B', representing the NES region), or
    - within a specified distance (default: 15Å) from any residue in that chain.
    
    Returns a list of residues to keep.
    """
    nes_residues = []
    other_residues = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' not in residue:
                    continue
                if chain.id == chain_id:
                    nes_residues.append(residue)
                else:
                    other_residues.append(residue)

    filtered_residues = nes_residues[:]
    for res in other_residues:
        res_coord = get_ca_coord(res)
        if any(np.linalg.norm(res_coord - get_ca_coord(nes_res)) < distance_threshold for nes_res in nes_residues):
            filtered_residues.append(res)

    return filtered_residues

class ResidueSelect(Select):
    """
    A helper class for PyMOL/Bio.PDB that allows writing only selected residues to a new PDB file.
    Used when saving a filtered structure.
    """
    def __init__(self, keep_residues):
        self.keep = {(res.get_parent().id, res.get_id()) for res in keep_residues}

    def accept_residue(self, residue):
        return (residue.get_parent().id, residue.get_id()) in self.keep

def save_filtered_structure(original_structure, residues_to_keep, output_file):
    """
    Save a filtered subset of residues from a PDB structure to a new PDB file.
    
    Args:
        original_structure: the full Bio.PDB structure
        residues_to_keep: a list of residues to include
        output_file: path to the output file
    """
    io = PDBIO()
    io.set_structure(original_structure)
    io.save(output_file, ResidueSelect(residues_to_keep))

def parse_pdb_to_graph(pdb_path, visualize=False, chain_id='B', save_filtered_pdb=False):
    """
    Parse a PDB file and convert the filtered structure into a PyTorch Geometric graph.
    
    Steps:
    - Filter the structure to include only chain B and nearby residues (within 15Å)
    - Represent each residue as a node with a one-hot feature vector + NES flag
    - Create edges between nodes based on 3D proximity (less than 8Å)
    - Optionally plot the graph in 2D (XY plane)

    Returns:
        torch_geometric.data.Data object
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    # Filter structure
    filtered_residues = filter_structure_by_chain_and_proximity(structure, chain_id)
    # Uncomment the next line to save the filtered structure to check if the cutoff works correctly
    filtered_file = "filtered_nes.pdb" 
    save_filtered_structure(structure, filtered_residues, filtered_file)

    # Parse filtered structure
    structure = parser.get_structure("filtered", filtered_file)
    if not save_filtered_pdb:
        os.remove(filtered_file)

    ca_coords = []
    node_features = []
    nes_flags = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca = residue['CA']
                    ca_coords.append(ca.coord)
                    one_hot = residue_to_one_hot(residue.get_resname())
                    # Create a flag for NES residues (chain B)
                    nes_flag = 1.0 if chain.id == chain_id else 0.0
                    
                    node_features.append(np.concatenate([one_hot, [nes_flag]]))
                    nes_flags.append(nes_flag)

    ca_coords = np.array(ca_coords)
    node_features = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(ca_coords, dtype=torch.float)
    num_nodes = len(ca_coords)

    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                if dist < DISTANCE_THRESHOLD:
                    edge_index.append([i, j])
                    edge_attr.append([dist])

    # Construct edges based on distance threshold
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    if visualize:
        plot_graph(data, nes_flags)

    return data

def plot_graph(graph, nes_flags):
    """
    Plot a 2D projection of the protein graph using matplotlib.
    - Red nodes represent NES residues (chain B)
    - Blue nodes represent neighboring residues
    - Edges represent proximity-based links
    """
    coords = graph.pos.numpy()
    colors = ['red' if flag else 'blue' for flag in nes_flags]

    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], s=50, c=colors, label='Residues')

    for edge in graph.edge_index.t().numpy():
        i, j = edge
        plt.plot([coords[i, 0], coords[j, 0]],
                 [coords[i, 1], coords[j, 1]],
                 color='gray', alpha=0.4)

    plt.title("Protein Graph (2D projection with NES coloring)")
    plt.xlabel("X (Å)")
    plt.ylabel("Y (Å)")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# Example usage
# pdb_file = "structures/af_positives/pos_1F71_6CD7_al.pdb"  # Replace with your actual file path
# graph = parse_pdb_to_graph(pdb_file, visualize=True)
# print(graph)
