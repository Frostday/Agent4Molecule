#!/usr/bin/env python
import argparse, sys, os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import SDWriter
from rdkit.Chem.MolStandardize import rdMolStandardize
from prep_sdf_util import prepare_to_sdf

def smiles_to_sdf(smiles: str, out_sdf: str, name: str = "LIG",
                  num_confs: int = 20, seed: int = 0) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    confs = AllChem.EmbedMultipleConfs(mol, numConfs=num_/jet/home/eshen3/Agent4Molecule/mcp_agent/util/clean_fragment.pyconfs, params=params)
    if not confs:
        raise RuntimeError("3D embedding failed")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=400)
        energies = [e for (_, e) in res]
    else:
        res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=400)
        energies = [e for (_, e) in res]

    best = int(min(range(len(confs)),
                   key=lambda i: 1e30 if energies[i] is None else energies[i]))

    os.makedirs(os.path.dirname(out_sdf) or ".", exist_ok=True)
    mol.SetProp("_Name", name)
    w = SDWriter(out_sdf)
    w.write(mol, confId=best)
    w.close()
    return out_sdf

def main():
    p = argparse.ArgumentParser(
        description="Prep SMILES or MOL/PDB → single-fragment SDF (H-explicit, 3D/minimized)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smiles", help="Input SMILES")
    g.add_argument("--infile", help="Input file (.mol or .pdb)")
    p.add_argument("--out", required=True, help="Output SDF path")
    p.add_argument("--name", default="LIG", help="Title stored in SDF")
    p.add_argument("--keep-salts", action="store_true",
                   help="Keep all fragments (default: keep only largest fragment)")
    p.add_argument("--no-gen3d", action="store_true",
                   help="Do NOT generate 3D (default: auto; SMILES=yes, files=infer)")
    args = p.parse_args()
    try:
        path = prepare_to_sdf(
            smiles=args.smiles,
            infile=args.infile,
            out_sdf=args.out,
            name=args.name,
            keep_salts=True if args.keep_salts else None,
            gen3d=False if args.no_gen3d else None
        )
        print(path)
        return 0
    except Exception as e:
        # Print errors to stderr so stdout stays clean
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
