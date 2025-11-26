RF_DIFFUSION_CONFIG = """
defaults:
  - aa

diffuser:
  T: {T_steps}

inference:
  num_designs: {N_designs}
  model_runner: NRBStyleSelfCond
  ligand: '{LIGAND}'

model:
  freeze_track_motif: True

contigmap:
  contigs: {residues}
  inpaint_str: null
  length: "100-150"

potentials:
  guiding_potentials: ["type:ligand_ncontacts,weight:1"] 
  guide_scale: 2
  guide_decay: cubic
"""


cst_template = """
CST::BEGIN

  TEMPLATE::   ATOM_MAP: 1 atom_name: {res1_atoms}
  TEMPLATE::   ATOM_MAP: 1 residue3:  {res1_residue}

  TEMPLATE::   ATOM_MAP: 2 atom_type: {res2_atoms}
  TEMPLATE::   ATOM_MAP: 2 residue3: {res2_residue}

  CONSTRAINT:: distanceAB:{distance_mean:8.1f}{distance_tol:7.2f}{distance_weight:7.1f}{distance_idxA:4d}{distance_idxB:4d}
  CONSTRAINT::    angle_A:{angleA_mean:8.1f}{angleA_tol:7.1f}{angleA_weight:7.1f}{angleA_periodicity:7.1f}{angleA_idx:2d}
  CONSTRAINT::    angle_B:{angleB_mean:8.1f}{angleB_tol:7.1f}{angleB_weight:7.1f}{angleB_periodicity:7.1f}{angleB_idx:2d}
  CONSTRAINT::  torsion_A:{torsA_mean:8.1f}{torsA_tol:7.1f}{torsA_weight:7.1f}{torsA_periodicity:7.1f}{torsA_idx:2d}
  CONSTRAINT:: torsion_AB:{torsAB_mean:8.1f}{torsAB_tol:7.1f}{torsAB_weight:7.1f}{torsAB_periodicity:7.1f}{torsAB_idx:2d}
  CONSTRAINT::  torsion_B:{torsB_mean:8.1f}{torsB_tol:7.1f}{torsB_weight:7.1f}{torsB_periodicity:7.1f}{torsB_idx:2d}

  ALGORITHM_INFO:: match
     MAX_DUNBRACK_ENERGY {max_dunbrack:.1f}
     IGNORE_UPSTREAM_PROTON_CHI
  ALGORITHM_INFO::END

CST::END
"""


def extract_job_id(output: str) -> str:
    """Extracts the job ID from the output of the sbatch command."""
    lines = output.split('\n')
    for line in lines:
        if "Submitted batch job" in line:
            return line.split()[-1]
    return ""
