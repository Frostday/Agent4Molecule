#!/usr/bin/env python3
import sys
from pathlib import Path

ATOM_PREFIX    = "ATOM  "
HETATM_PREFIX  = "HETATM"
TER_PREFIX     = "TER   "
HEADER_PREFIXES = (
    "HEADER", "TITLE ", "REMARK", "COMPND", "SOURCE",
    "EXPDTA", "KEYWDS", "AUTHOR", "JRNL", "DBREF",
    "SEQRES", "HET   ", "HETNAM", "FORMUL", "CRYST1",
    "ORIGX", "SCALE", "MTRIX", "MASTER"
)
MODEL_PREFIX   = "MODEL "
ENDMDL_PREFIX  = "ENDMDL"

def main():
    if len(sys.argv) != 4:
        print("Usage: python sep_protein_ligand_min.py <combined.pdb> <output_protein_path> <output_ligand_path>")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out_pro = Path(sys.argv[2])
    out_lig = Path(sys.argv[3])

    if not inp.exists():
        print(f"Error: file not found: {inp}")
        sys.exit(1)

    lines = inp.read_text().splitlines(True)

    pro, lig = [], []

    # Copy headers & model markers to both
    for ln in lines:
        if ln.startswith(HEADER_PREFIXES) or ln.startswith(MODEL_PREFIX) or ln.startswith(ENDMDL_PREFIX):
            pro.append(ln)
            lig.append(ln)

    last_dest = None
    for ln in lines:
        rec = ln[:6]
        if rec == ATOM_PREFIX:
            pro.append(ln); last_dest = "pro"
        elif rec == HETATM_PREFIX:
            lig.append(ln); last_dest = "lig"
        elif rec == TER_PREFIX:
            (pro if last_dest == "pro" else lig if last_dest == "lig" else pro).append(ln)

    pro.append("END\n")
    lig.append("END\n")

    out_pro.parent.mkdir(parents=True, exist_ok=True)
    out_lig.parent.mkdir(parents=True, exist_ok=True)
    out_pro.write_text("".join(pro))
    out_lig.write_text("".join(lig))

    print(f"Wrote protein -> {out_pro}")
    print(f"Wrote ligand  -> {out_lig}")

if __name__ == "__main__":
    main()
