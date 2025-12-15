from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import sys

inp, outp = sys.argv[1], sys.argv[2]
sup = Chem.SDMolSupplier(inp, removeHs=False)
best = None; best_atoms = -1

for m in sup:
    if m is None: 
        continue
    m = rdMolStandardize.LargestFragmentChooser().choose(m)  # drop disconnected bits
    n = m.GetNumAtoms()
    if n > best_atoms:
        best = m; best_atoms = n

if best is None:
    raise SystemExit("No valid molecules found")

# sanity: ensure one fragment
if len(Chem.GetMolFrags(best)) != 1:
    raise SystemExit("Still >1 fragment after cleanup")

w = Chem.SDWriter(outp)
w.write(best)
w.close()
print(outp)
