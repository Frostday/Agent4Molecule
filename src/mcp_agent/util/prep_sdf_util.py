import os
from typing import Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

def has_3d(mol: Chem.Mol) -> bool:
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer(0)
    for i in range(mol.GetNumAtoms()):
        if abs(conf.GetAtomPosition(i).z) > 1e-3:
            return True
    return False

def infer_policy_from_mol(mol: Chem.Mol) -> Tuple[bool, bool]:
    # Default policy: drop salts (keep only largest fragment).
    keep_salts = False
    # Generate 3D only if the molecule lacks 3D coords.
    gen3d = not has_3d(mol)
    return keep_salts, gen3d

def load_mol_auto(path: str, *, sanitize: bool = True) -> Chem.Mol:
    """Load first valid molecule from .mol."""
    ext = os.path.splitext(path.lower())[1]
    if ext == ".mol":
        m = Chem.MolFromMolFile(path, removeHs=False, sanitize=sanitize)
        if m is None:
            raise ValueError(f"Failed to read MOL: {path}")
        return m
    raise ValueError(f"Unsupported input format: {ext} (use .mol)")

def prepare_to_sdf(
    *,
    smiles: Optional[str] = None,
    infile: Optional[str] = None,
    out_sdf: str,
    name: str = "LIG",
    keep_salts: Optional[bool] = None,
    gen3d: Optional[bool] = None
) -> str:
    """
    Convert SMILES or MOL to a SINGLE-fragment SDF with explicit H, optional 3D + minimization.

    - If keep_salts is None: default is to DROP salts (largest fragment only).
    - If gen3d is None: SMILES => True; files => infer from presence of 3D.
    - For PDB inputs: 
    """
    assert (smiles is None) ^ (infile is None), "Provide exactly one of smiles or infile"

    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        if keep_salts is None:
            keep_salts = False
        if not keep_salts:
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        gen3d = True if gen3d is None else gen3d
    else:
        mol = load_mol_auto(infile, sanitize=True)
        ext = os.path.splitext(infile.lower())[1]
        inf_keep, inf_gen3d = infer_policy_from_mol(mol)
        if keep_salts is None:
            keep_salts = inf_keep
        if gen3d is None:
            gen3d = inf_gen3d
        if not keep_salts:
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)

    # Add explicit H for correct chemistry & geometry
    mol = Chem.AddHs(mol)

    best_conf = 0
    if gen3d or mol.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=max(1, 20), params=params)
        if not cids:
            raise RuntimeError("3D embedding failed")

        if AllChem.MMFFHasAllMoleculeParams(mol):
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=400)
            energies = [e for (_, e) in res]
        else:
            res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=400)
            energies = [e for (_, e) in res]
        best_conf = int(min(range(len(cids)), key=lambda i: 1e30 if energies[i] is None else energies[i]))
    else:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=400)

    mol.SetProp("_Name", name)
    os.makedirs(os.path.dirname(out_sdf) or ".", exist_ok=True)
    w = Chem.SDWriter(out_sdf); w.write(mol, confId=best_conf); w.close()
    return os.path.abspath(out_sdf)
