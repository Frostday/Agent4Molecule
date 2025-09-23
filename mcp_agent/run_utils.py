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
  contigs: ["30-110,A15-15,30-110"]
  inpaint_str: null
  length: "100-140"

potentials:
  guiding_potentials: ["type:ligand_ncontacts,weight:1"] 
  guide_scale: 2
  guide_decay: cubic
"""

def extract_job_id(output: str) -> str:
    """Extracts the job ID from the output of the sbatch command."""
    lines = output.split('\n')
    for line in lines:
        if "Submitted batch job" in line:
            return line.split()[-1]
    return ""
