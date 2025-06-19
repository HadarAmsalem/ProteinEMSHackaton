import os
from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB.Polypeptide import PPBuilder
import matplotlib.pyplot as plt
from esm_embeddings import get_esm_model, get_esm_embeddings

# === Constants ===
DISTANCE_THRESHOLD = 8.0
NES_PROXIMITY_THRESHOLD = 15.0
STANDARD_AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}


def residue_to_one_hot(resname):
    """
    Convert a residue name to a one-hot encoded vector.
    param resname: Residue name (e.g., 'ALA', 'CYS', etc.)
    return: One-hot encoded vector of shape (20,)
    """
    vec = np.zeros(len(STANDARD_AMINO_ACIDS), dtype=np.float32)
    if resname in AA_TO_INDEX:
        vec[AA_TO_INDEX[resname]] = 1.0
    return vec


def get_ca_coord(residue):
    """
    Get the coordinates of the alpha carbon (CA) atom in a residue.
    """
    return residue['CA'].coord if 'CA' in residue else None


def filter_structure_by_chain_and_proximity(structure, chain_id='B', distance_threshold=NES_PROXIMITY_THRESHOLD):
    """
    Filter the structure to keep only residues from a specific chain and those within a certain distance
    from the alpha carbon of residues in that chain.
    param structure: Bio.PDB Structure object
    param chain_id: Chain ID to filter by (default is 'B' for NES)
    param distance_threshold: Distance threshold for proximity filtering (default is 15.0 Å)
    return: List of filtered residues
    """
    nes_residues, other_residues = [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' not in residue:
                    continue
                (nes_residues if chain.id == chain_id else other_residues).append(residue)

    filtered_residues = nes_residues[:]
    for res in other_residues:
        res_coord = get_ca_coord(res)
        if any(np.linalg.norm(res_coord - get_ca_coord(nes_res)) < distance_threshold for nes_res in nes_residues):
            filtered_residues.append(res)
    return filtered_residues


class ResidueSelect(Select):
    """
    Custom Select class to filter residues based on a set of residues to keep.
    This class is used with PDBIO to save only the specified residues.
    """
    def __init__(self, keep_residues):
        self.keep = {(res.get_parent().id, res.get_id()) for res in keep_residues}

    def accept_residue(self, residue):
        return (residue.get_parent().id, residue.get_id()) in self.keep


def save_filtered_structure(original_structure, residues_to_keep, output_file):
    """
    Save the filtered structure to a PDB file, keeping only the specified residues.
    """
    io = PDBIO()
    io.set_structure(original_structure)
    io.save(output_file, ResidueSelect(residues_to_keep))


def parse_pdb_to_graph(pdb_path, visualize=False, chain_id='B', save_filtered_pdb=False,
                       encoding_type = 'esm' , esm_embedding_size=1260,
                       esm_layer=30):
    """
    Parse a PDB file and convert it into a PyTorch Geometric Data object.
    param pdb_path: Path to the PDB file
    param visualize: Whether to visualize the graph (default is False)
    param chain_id: Chain ID to filter by (default is 'B' for NES)
    param save_filtered_pdb: Whether to save the filtered PDB file (default is False)
    param encoding_type: Type of encoding for node features ('esm' or 'onehot')
    param esm_embedding_size: Size of the ESM embedding (default is 1260)
    param esm_layer: ESM layer to use for embeddings (default is 30)
    return: PyTorch Geometric Data object representing the protein graph
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    filtered_residues = filter_structure_by_chain_and_proximity(structure, chain_id)
    filtered_file = "filtered_nes.pdb"
    save_filtered_structure(structure, filtered_residues, filtered_file)
    structure = parser.get_structure("filtered", filtered_file)
    if not save_filtered_pdb:
        os.remove(filtered_file)

    ca_coords, nes_flags, residue_list, aa_seq = [], [], [], ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].coord)
                    nes_flags.append(1.0 if chain.id == chain_id else 0.0)
                    residue_list.append(residue)

    ca_coords = np.array(ca_coords)
    pos = torch.tensor(ca_coords, dtype=torch.float)

    if encoding_type == 'esm':
        ppb = PPBuilder()
        seq = str(ppb.build_peptides(structure[0][chain_id])[0].get_sequence())
        esm_model, alphabet, batch_converter, device = get_esm_model(embedding_size=esm_embedding_size)
        embeddings = get_esm_embeddings([("protein", seq)], esm_model, alphabet, batch_converter, device,
                                        embedding_layer=esm_layer, sequence_embedding=False)[0]
        node_features = torch.tensor(embeddings, dtype=torch.float)
    elif encoding_type == 'onehot':
        node_features = []
        for residue, flag in zip(residue_list, nes_flags):
            one_hot = residue_to_one_hot(residue.get_resname())
            node_features.append(np.concatenate([one_hot, [flag]]))
        node_features = torch.tensor(node_features, dtype=torch.float)

    edge_index, edge_attr = [], []
    for i in range(len(ca_coords)):
        for j in range(len(ca_coords)):
            if i != j:
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                if dist < DISTANCE_THRESHOLD:
                    edge_index.append([i, j])
                    edge_attr.append([dist])

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
