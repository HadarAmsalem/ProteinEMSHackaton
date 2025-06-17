import os
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform

# Parse only C-alpha atoms from PDB
def parse_ca_atoms(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    atom = residue['CA']
                    ca_atoms.append((atom.get_serial_number(), atom.coord, chain.id))
    return ca_atoms

# Convert to graph
def build_graph_from_atoms(atoms, distance_threshold=10.0):
    G = nx.Graph()
    coords = np.array([coord for _, coord, _ in atoms])
    for i, (_, coord, chain) in enumerate(atoms):
        G.add_node(i, chain=chain, pos=coord)
    dists = squareform(pdist(coords))
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if dists[i, j] < distance_threshold:
                G.add_edge(i, j, weight=dists[i, j])
    return G

# Folder with your PDB files
pdb_folder = Path(".\\structures\\structures\\af_negatives")  # your folder path

# Search recursively in subfolders
pdb_files = list(pdb_folder.rglob("*.pdb"))
print(f"Found {len(pdb_files)} PDB files")

graphs = []
for pdb_file in pdb_files:
    atoms = parse_ca_atoms(pdb_file)
    G = build_graph_from_atoms(atoms)
    graphs.append((pdb_file.name, G))

# Visualize first one
if graphs:
    name, G = graphs[0]
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_size=30, node_color='skyblue', edge_color='gray')
    plt.title(f"Graph view of {name}")
    plt.axis('off')
    plt.show()
else:
    print("No graphs were created. Check file contents.")
