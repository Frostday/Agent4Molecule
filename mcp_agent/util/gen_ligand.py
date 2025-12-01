import argparse
from openff.toolkit.topology import Molecule

def generate_ligand_pdb(input_sdf_path, output_pdb_path):
    """Generate ligand PDB file from SDF using OpenFF."""
    # Load your ligand from SDF
    mol = Molecule.from_file(input_sdf_path)

    # Save to PDB
    mol.to_file(output_pdb_path, "PDB")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate ligand PDB file from SDF using OpenFF.")
    parser.add_argument("-i", "--input", required=True, help="Path to input ligand SDF file.")
    parser.add_argument("-o", "--output", required=True, help="Path to output ligand PDB file.")
    args = parser.parse_args()
    generate_ligand_pdb(args.input, args.output)
