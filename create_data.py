from pathlib import Path
from Bio.PDB import PDBParser, PPBuilder
import pandas as pd

# === Configuration ===
# Parent folder where the af_positives and af_negatives are located
data_root = Path(".\\structures\\structures")  # change this!
label_map = {
    "af_positives": 1,
    "af_negatives": 0
}

parser = PDBParser(QUIET=True)
ppb = PPBuilder()

entries = []

# Walk through both positive and negative folders
for subfolder, label in label_map.items():
    folder = data_root / subfolder
    for pdb_file in folder.glob("*.pdb"):
        try:
            structure = parser.get_structure(pdb_file.stem, pdb_file)
            # extract sequence (assume peptide is in chain B or the shortest chain)
            all_seqs = []
            for model in structure:
                for chain in model:
                    peptides = ppb.build_peptides(chain)
                    for peptide in peptides:
                        seq = str(peptide.get_sequence())
                        all_seqs.append(seq)
            if all_seqs:
                # Take the longest sequence (likely the peptide)
                final_seq = max(all_seqs, key=len)
                entries.append({
                    "id": pdb_file.stem,
                    "sequence": final_seq,
                    "label": label
                })
        except Exception as e:
            print(f"⚠️ Skipping {pdb_file.name}: {e}")

# Save to CSV
df = pd.DataFrame(entries)
df.to_csv("protein_nes_dataset.csv", index=False)
print("✅ CSV saved as protein_nes_dataset.csv")
