from Bio.PDB import PDBParser, PDBIO 
from Bio.PDB.Chain import Chain 
from Bio.PDB.Residue import Residue 
from Bio.PDB.Atom import Atom 
import numpy as np 
import sys 
import argparse
from pathlib import Path

def merge_receptor_ligand_pdbqt(receptor_pdbqt_path: str,
                                ligand_pdbqt_path: str,
                                out_pdbqt_path: str,
                                ligand_chain: str = "L",
                                ligand_resname: str = "HBA") -> None:
    """
    Merge receptor (PDBQT) and ligand (multi-pose PDBQT) into a complex PDBQT.

    - Receptor kept as-is (minus trailing END/TER/ENDMDL).
    - Ligand: takes ATOM/HETATM lines from MODEL 1 (or the first MODEL block).
    - Ligand atoms are written as HETATM on chain `ligand_chain`, residue name `ligand_resname`, resSeq = 1.
    - Ligand atom serials continue after receptor's max serial.
    - PDBQT extras (partial charge/type) preserved.
    """

    def read_lines(path):
        with open(path, "r") as f:
            return f.readlines()

    def strip_trailing_terminators(lines):
        while lines and lines[-1].strip().upper() in {"END", "TER", "ENDMDL"}:
            lines.pop()
        return lines

    def is_atom_line(ln: str) -> bool:
        s = ln.lstrip()
        return s.startswith("ATOM") or s.startswith("HETATM")

    def find_existing_chains(lines):
        chains = set()
        for ln in lines:
            if is_atom_line(ln) and len(ln) >= 22:
                chains.add(ln[21])
        return chains

    def max_serial(lines):
        m = 0
        for ln in lines:
            if is_atom_line(ln):
                try:
                    serial = int(ln[6:11])
                    m = max(m, serial)
                except ValueError:
                    pass
        return m

    def get_first_model_block(lines):
        """Return the slice (list of lines) between the first 'MODEL' and its 'ENDMDL'."""
        start_idx = None
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("MODEL"):
                start_idx = i
                break
        if start_idx is None:
            # No MODEL/ENDMDL: take the first contiguous block of ATOM/HETATM
            atoms = []
            started = False
            for ln in lines:
                if is_atom_line(ln):
                    atoms.append(ln)
                    started = True
                elif started:
                    break
            return atoms
        # find ENDMDL
        end_idx = None
        for j in range(start_idx + 1, len(lines)):
            if lines[j].lstrip().startswith("ENDMDL"):
                end_idx = j
                break
        block = lines[start_idx:end_idx] if end_idx is not None else lines[start_idx:]
        return block

    def extract_atom_lines_from_block(block_lines):
        """From a MODEL block (with ROOT/BRANCH/etc.), return only ATOM/HETATM lines."""
        return [ln for ln in block_lines if is_atom_line(ln)]

    def next_chain_id(preferred, used):
        if preferred not in used:
            return preferred
        c = preferred
        while True:
            c = chr(ord(c) + 1)
            if c > 'Z':
                c = 'A'
            if c not in used:
                return c

    def rewrite_ligand_atom_line(ln, new_serial, chain_id, resname, resseq=1):
        """
        Edit PDB fixed-width fields in a PDBQT line while preserving the tail (charges/types).
        PDB columns (1-based):
          1-6  record name
          7-11 serial
         18-20 resName
            22 chainID
         23-26 resSeq
        """
        if len(ln) < 80:
            ln = ln.rstrip("\n") + " " * (80 - len(ln)) + "\n"

        buf = list(ln)
        # Force record to HETATM
        buf[0:6]  = list("HETATM")
        # Serial (right-aligned in width 5)
        buf[6:11] = list(f"{new_serial:5d}")
        # resName (right-aligned width 3)
        buf[17:20] = list(f"{resname:>3s}"[:3])
        # chain ID
        buf[21] = chain_id[0]
        # resSeq (right-aligned width 4)
        buf[22:26] = list(f"{resseq:4d}")
        return "".join(buf)

    # ---- receptor ----
    rec_lines = read_lines(receptor_pdbqt_path)
    rec_lines = strip_trailing_terminators(rec_lines)
    used_chains = find_existing_chains(rec_lines)
    chain_id = next_chain_id(ligand_chain, used_chains)
    start_serial = max_serial(rec_lines)

    # ---- ligand (first model) ----
    lig_lines_all = read_lines(ligand_pdbqt_path)
    first_model_block = get_first_model_block(lig_lines_all)
    lig_atom_lines = extract_atom_lines_from_block(first_model_block)
    if not lig_atom_lines:
        raise ValueError(
            f"No ATOM/HETATM lines found in ligand PDBQT first model: {ligand_pdbqt_path}\n"
            f"Tip: try `vina_split --input {ligand_pdbqt_path}` and pass the *_ligand_1.pdbqt file."
        )

    # ---- merge ----
    merged = list(rec_lines)
    serial = start_serial
    for ln in lig_atom_lines:
        serial += 1
        merged.append(rewrite_ligand_atom_line(ln, serial, chain_id, ligand_resname, resseq=1))

    merged.append("END\n")

    with open(out_pdbqt_path, "w") as f:
        f.writelines(merged)


def main():
    """Command-line interface for merging receptor and ligand PDBQT files."""
    parser = argparse.ArgumentParser(
        description="Merge receptor (PDBQT) and ligand (multi-pose PDBQT) into a complex PDBQT file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python combine_protein_ligand.py -r receptor.pdbqt -l ligand.pdbqt -o complex.pdbqt
  python combine_protein_ligand.py -r receptor.pdbqt -l ligand.pdbqt -o complex.pdbqt -c M -n HEM
        """
    )
    
    parser.add_argument(
        "-r", "--receptor",
        required=True,
        help="Path to receptor PDBQT file"
    )
    parser.add_argument(
        "-l", "--ligand",
        required=True,
        help="Path to ligand PDBQT file (multi-pose or single pose)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output merged PDBQT file"
    )
    parser.add_argument(
        "-c", "--chain",
        default="L",
        help="Chain ID for the ligand (default: L)"
    )
    parser.add_argument(
        "-n", "--resname",
        default="HBA",
        help="Residue name for the ligand (default: HBA)"
    )
    
    args = parser.parse_args()
    
    try:
        merge_receptor_ligand_pdbqt(
            receptor_pdbqt_path=args.receptor,
            ligand_pdbqt_path=args.ligand,
            out_pdbqt_path=args.output,
            ligand_chain=args.chain,
            ligand_resname=args.resname
        )
        print(f"Successfully merged receptor and ligand into: {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
    
    # Example usage (commented out):
    # merge_receptor_ligand_pdbqt(
    #     receptor_pdbqt_path="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output/inputs/receptor_output.pdbqt",
    #     ligand_pdbqt_path="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output/inputs/docked.pdbqt",
    #     out_pdbqt_path="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output/inputs/complex.pdbqt",
    # )
