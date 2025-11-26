from rdkit import Chem
from rdkit.Chem import AllChem

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--infile", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

mol = Chem.MolFromMolFile(args.infile, sanitize=False, removeHs=False)
if mol is None:
    raise ValueError("Failed to read molecule")

# Remove atoms that are queries or dummy placeholders (e.g., 'R' or '*')
to_remove = []
for atom in mol.GetAtoms():
    sym = atom.GetSymbol()
    if sym in ("R", "*") or atom.HasQuery():
        to_remove.append(atom.GetIdx())

if to_remove:
    em = Chem.EditableMol(mol)
    for idx in sorted(to_remove, reverse=True):
        em.RemoveAtom(idx)
    mol = em.GetMol()

# Sanitize and add explicit hydrogens
Chem.SanitizeMol(mol)
mol = Chem.AddHs(mol)

# Generate 3D coordinates
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

# Write docking-ready SDF
w = Chem.SDWriter(args.out)
w.write(mol)
w.close()