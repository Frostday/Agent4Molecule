import argparse
from openmm.app import PDBFile, Modeller
from openmm.app import ForceField

def merge_protein_ligand(protein_path, ligand_path, output_path):
    protein = PDBFile(protein_path)      # AF2 or Enzygen or RFDiffusion output
    ligand  = PDBFile(ligand_path)       # Converted from OpenFF SDF

    # Put ligand in a different chain
    for atom in ligand.topology.atoms():
        if atom.residue.name != "LIG":
            atom.residue.name = "LIG"

    # Merge with modeller
    modeller = Modeller(protein.topology, protein.positions)
    modeller.add(ligand.topology, ligand.positions)

    # Save complex
    with open(output_path, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge protein and ligand PDB files into a complex PDB file.")
    parser.add_argument("-p", "--protein", required=True, help="Path to protein PDB file.")
    parser.add_argument("-l", "--ligand", required=True, help="Path to ligand PDB file.")
    parser.add_argument("-o", "--output", required=True, help="Path to output complex PDB file.")
    args = parser.parse_args()

    merge_protein_ligand(args.protein, args.ligand, args.output)
