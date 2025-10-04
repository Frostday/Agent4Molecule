from typing import List, Dict, Tuple, Optional
from typing_extensions import Annotated
from pydantic import Field

from collections import Counter
from Bio import AlignIO
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa

import numpy as np

AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWYOUX")
THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","SEC":"U","PYL":"O"
}

def _three_to_one_safe(resname: str) -> str:
    return THREE_TO_ONE.get(resname.upper(), "X")

def _conserved_columns(aln, tau: float) -> List[Tuple[int, str, float]]:
    """List of (msa_col, consensus_aa, frac) where frac >= tau."""
    L = aln.get_alignment_length()
    out = []
    for j in range(L):
        col = [rec.seq[j] for rec in aln]
        aa = [c for c in col if c != "-" and c.upper() in AA_LETTERS]
        if not aa:
            continue
        cnt = Counter(aa)
        most, f = cnt.most_common(1)[0]
        frac = f / len(aa)
        if frac >= tau:
            out.append((j, most, frac))
    return out

def _build_msa_to_ungapped_maps(aln):
    """Return list (per MSA col) of dict: seq_id -> ungapped_idx (0-based)."""
    L = aln.get_alignment_length()
    ids = [rec.id for rec in aln]
    ungapped = [0] * len(aln)
    maps = []
    for j in range(L):
        m = {}
        for i, rec in enumerate(aln):
            c = rec.seq[j]
            if c != "-":
                m[ids[i]] = ungapped[i]
                ungapped[i] += 1
        maps.append(m)
    return maps

def _load_chain(struct_path: str, chain_id: str):
    parser = MMCIFParser(QUIET=True) if struct_path.lower().endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    structure = parser.get_structure("struct", struct_path)
    model = structure[0]
    if chain_id not in model:
        raise ValueError(f"Chain '{chain_id}' not in structure. Available: {[ch.id for ch in model]}")
    return model[chain_id]

def _chain_seq_and_reslist(chain):
    """Return (1-letter seq, list of AA residues) for standard amino acids only."""
    seq = []
    reslist = []
    for res in chain.get_residues():
        if not is_aa(res, standard=True):
            continue
        seq.append(_three_to_one_safe(res.get_resname()))
        reslist.append(res)
    return "".join(seq), reslist

def _map_ref_to_chain_coords(
    ref_seq: str,
    chain_seq: str,
    reslist,                      # list of Biopython Residue objects for the chain (in order)
    ref_indices_0: List[int],     # ungapped 0-based indices on the ref sequence to map
) -> Dict[int, Optional[List[float]]]:
    """
    Map ungapped ref_seq indices -> chain residue Cα coordinates using a global alignment.
    Works with Bio.Align.PairwiseAligner (no seqA/seqB fields).
    """
    # Global alignment
    aligner = PairwiseAligner()
    aligner.mode = "global"
    # (optional) scoring tweaks similar-ish to your pairwise2 setup
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    alns = aligner.align(ref_seq, chain_seq)
    if len(alns) == 0:
        # fall back: naive one-to-one if equal length
        mapping = {i: i for i in range(min(len(ref_seq), len(chain_seq)))}
    else:
        best = alns[0]
        # best.aligned is a tuple (ref_blocks, chain_blocks)
        # each is an array of shape (nblocks, 2) with [start, end) coordinates
        ref_blocks = np.asarray(best.aligned[0])
        chain_blocks = np.asarray(best.aligned[1])

        mapping: Dict[int, int] = {}
        for (r_start, r_end), (c_start, c_end) in zip(ref_blocks, chain_blocks):
            # within each aligned block, positions correspond 1-to-1
            block_len = (r_end - r_start)
            # safety: Biopython guarantees equal lengths for aligned blocks
            for off in range(block_len):
                mapping[r_start + off] = c_start + off

    out: Dict[int, Optional[List[float]]] = {}
    for r_idx in ref_indices_0:
        c_idx = mapping.get(r_idx, None)
        if c_idx is None or c_idx < 0 or c_idx >= len(reslist):
            out[r_idx] = None
            continue
        res = reslist[c_idx]
        ca = res["CA"] if "CA" in res else None
        if ca is None:
            out[r_idx] = None
        else:
            x, y, z = ca.get_coord()
            out[r_idx] = [float(x), float(y), float(z)]
    return out


def msa_to_enzygen_motif(
    aln_path: str,
    aln_format: str,
    structure_path: str,
    chain_id: str,
    *,
    ref_id: Optional[str] = None,
    tau: float = 0.67,
    idx_base: int = 0,
    drop_unmapped: bool = True,
) -> Dict[
    str,
    Annotated[
        List,  # concrete types are in the signatures below; dict keys are fixed
        Field(description="motif fields for EnzyGen")
    ]
]:
    """
    Build EnzyGen motif fields from a ClustalW MSA and a representative PDB chain.

    Returns exactly these keys:
      - motif_indices: Annotated[list[int], Field(description="Indices of the motif")]
      - motif_seq:     Annotated[list[str], Field(description="Sequence of the motif")]
      - motif_coord:   Annotated[list[list[float]], Field(description="Coordinates of the motif")]

    Notes:
      * Indices refer to the ungapped reference sequence (0- or 1-based via idx_base).
      * By default, sites that cannot be mapped to Cα coordinates are dropped (drop_unmapped=True).
        Set drop_unmapped=False to keep them with None coordinates (may violate your schema).
    """
    # Load alignment
    aln = AlignIO.read(aln_path, aln_format)
    ref_rec = aln[0] if ref_id is None else next((r for r in aln if r.id == ref_id), None)
    if ref_rec is None:
        raise ValueError(f"Reference id '{ref_id}' not found. Available: {[r.id for r in aln]}")

    # Pick conserved positions
    important = _conserved_columns(aln, tau)
    msa2ungapped = _build_msa_to_ungapped_maps(aln)

    # Extract reference sites (ungapped indices)
    sites = []
    for j, aa, frac in important:
        c = ref_rec.seq[j]
        if c != "-":
            idx0 = msa2ungapped[j][ref_rec.id]  # 0-based
            sites.append((idx0, str(c)))

    # Load structure and build chain sequence
    chain = _load_chain(structure_path, chain_id)
    chain_seq, reslist = _chain_seq_and_reslist(chain)

    # Align ungapped reference to chain and map coordinates
    ref_ungapped = str(ref_rec.seq).replace("-", "")
    ref_indices_0 = [idx0 for idx0, _ in sites]
    coords_map = _map_ref_to_chain_coords(ref_ungapped, chain_seq, reslist, ref_indices_0)

    # Compose outputs
    motif_indices: List[int] = []
    motif_seq:     List[str] = []
    motif_coord:   List[List[float]] = []

    for idx0, aa in sites:
        coord = coords_map.get(idx0)
        if coord is None and drop_unmapped:
            continue
        if coord is None and not drop_unmapped:
            # This will violate your strict type if you enforce non-null;
            # consider using drop_unmapped=True (default).
            continue
        motif_indices.append(idx0 + (1 if idx_base == 1 else 0))
        motif_seq.append(aa)
        motif_coord.append(coord)  # type: ignore[arg-type]

    return {
        "motif_indices": motif_indices,  # Annotated[list[int], Field(...)]
        "motif_seq": motif_seq,          # Annotated[list[str], Field(...)]
        "motif_coord": motif_coord,      # Annotated[list[list[float]], Field(...)]
    }
