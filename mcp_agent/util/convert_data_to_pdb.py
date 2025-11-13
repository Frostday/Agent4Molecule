import json
import numpy as np

def write_ca_pdb(sequence, coords, outfile="output.pdb"):
    """
    sequence: string of 1-letter amino acids (e.g., 'ACDE...')
    coords:   list of (x, y, z) floats, one per residue
    """

    aa_map = {
        'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE',
        'G':'GLY','H':'HIS','I':'ILE','K':'LYS','L':'LEU',
        'M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG',
        'S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR'
    }

    with open(outfile, "w") as f:
        atom_id = 1
        for i, (aa, (x, y, z)) in enumerate(zip(sequence, coords), start=1):
            resname = aa_map.get(aa, "UNK")
            line = ("ATOM  {atom_id:5d}  CA  {resname:3s} A{resnum:4d}    "
                    "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n").format(
                atom_id=atom_id,
                resname=resname,
                resnum=i,
                x=x, y=y, z=z,
            )
            f.write(line)
            atom_id += 1

        f.write("END\n")


data_file = "/ocean/projects/cis240137p/dgarg2/github/PPDiff/data/sample_data_binder_design.json"
with open(data_file, "r") as f:
    data = json.load(f)
seq = data["binder_design"]["test"]["seqs"][0]
coords = data["binder_design"]["test"]["coors"][0]
coords = [coors[1] for coors in coords]
total_len = len(data["binder_design"]["test"]["target"][0])
binder_len = sum(np.array(data["binder_design"]["test"]["target"][0])==0)
target_len = total_len - binder_len
print(total_len, binder_len, target_len)
seq = seq[:target_len]
coords = coords[:target_len]
write_ca_pdb(seq, coords, "/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/inputs/binder_design_example.pdb")

data_file = "/ocean/projects/cis240137p/dgarg2/github/PPDiff/data/sample_data_antibody_design_cdrh1.json"
with open(data_file, "r") as f:
    data = json.load(f)
seq = data["antibody_design"]["test"]["seqs"][0]
coords = data["antibody_design"]["test"]["coors"][0]
print(len(data["antibody_design"]["test"]["target"][0]), np.where(np.array(data["antibody_design"]["test"]["target"][0])==0))
write_ca_pdb(seq, coords, "/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/inputs/antibody_design_example_1.pdb")
