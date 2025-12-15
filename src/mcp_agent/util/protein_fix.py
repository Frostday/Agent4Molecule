import argparse
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def fix_protein(input_pdb_path, output_pdb_path):
    """Fix protein PDB file using PDBFixer."""
    fixer = PDBFixer(filename=input_pdb_path)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    with open(output_pdb_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix protein PDB file using PDBFixer.")
    parser.add_argument("-i", "--input", required=True, help="Path to input protein PDB file.")
    parser.add_argument("-o", "--output", required=True, help="Path to output fixed protein PDB file.")
    args = parser.parse_args()
    fix_protein(args.input, args.output)
