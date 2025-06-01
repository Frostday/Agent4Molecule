from mcp.server.fastmcp import FastMCP
from typing import Any,Annotated
import httpx
import os,sys,glob
from pydantic import Field
sys.path.append("/ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/scripts/utils")
import utils
import getpass
import subprocess
import time
import importlib
from shutil import copy2



mcp = FastMCP("heme-binder")


@mcp.tool()
def analyze_diffusion_output(script_dir: Annotated[str, Field(description="File provided for user")]):
    SCRIPT_DIR = os.path.dirname(script_dir)  # edit this to the GitHub repo path. Throws an error by default.
    if not os.path.exists(SCRIPT_DIR):
        raise RuntimeError(f"Missing SCRIPT_DIR: {SCRIPT_DIR}")
    sys.path.append(SCRIPT_DIR+"/scripts/utils")
    diffusion_script = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/rf_diffusion_all_atom/run_inference.py"
    proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"

    CONDAPATH = "/ocean/projects/cis240137p/ksubram4/anaconda3"   # edit this depending on where your Conda environments live
    PYTHON = {"diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
          "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
          "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
          "general": f"{CONDAPATH}/envs/diffusion/bin/python"}
    
    WDIR = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/outputs"

    if not os.path.exists(WDIR):
        os.makedirs(WDIR, exist_ok=True)

    print(f"Working directory: {WDIR}")

    USE_GPU_for_AF2 = True
    params = [f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"]  # Rosetta params file(s)
    LIGAND = "HBA"

    diffusion_inputs = glob.glob(f"{SCRIPT_DIR}/input/*.pdb")
    print(f"Found {len(diffusion_inputs)} PDB files")


    DIFFUSION_DIR = f"{WDIR}/0_diffusion"
    if not os.path.exists(DIFFUSION_DIR):
        os.makedirs(DIFFUSION_DIR, exist_ok=False)

    os.chdir(DIFFUSION_DIR)

    diffusion_rundirs = ['7o2g_HBA']
    analysis_script = f"{SCRIPT_DIR}/scripts/diffusion_analysis/process_diffusion_outputs.py"

    diffusion_outputs = []
    for d in diffusion_rundirs:
        diffusion_outputs += glob.glob(f"{d}/out/*.pdb")
    
    analysis_script = f"{SCRIPT_DIR}/scripts/diffusion_analysis/process_diffusion_outputs.py"



# By default I don't use the --analyze flag. As a result the backbones are filtered as the script runs.
# You can set --analyze to True to calculate all scores for all backbones.
# This will slow the analysis down, but you can then filter the backbones separately afterwards.
    dif_analysis_cmd_dict = {"--pdb": " ".join(diffusion_outputs),
                        # "--ref": f"{SCRIPT_DIR}/input/*.pdb",
                        "--ref": f"{SCRIPT_DIR}/input/7o2g_HBA.pdb",
                        "--params": " ".join(params),
                        "--term_limit": "15.0",
                        "--SASA_limit": "0.3",  # Highest allowed relative SASA of ligand
                        "--loop_limit": "0.4",  # Fraction of backbone that can be loopy
                        "--ref_catres": "A15",  # Position of CYS in diffusion input
                        "--rethread": True,
                        "--fix": True,
                        "--exclude_clash_atoms": "O1 O2 O3 O4 C5 C10",  # Ligand atoms excluded from clashchecking because they are flexible
                        "--ligand_exposed_atoms": "C45 C46 C47",  # Ligand atoms that need to be more exposed
                        "--exposed_atom_SASA": "10.0",  # minimum absolute SASA for exposed ligand atoms
                        "--longest_helix": "30",
                        "--rog": "30.0",
                        "--partial": None,
                        "--outdir": None,
                        "--traj": "5/30",  # Also random 5 models are taken from the last 30 steps of the diffusion trajectory
                        "--trb": None,
                        "--analyze": False,
                        "--nproc": "2"}

    analysis_command = f"{PYTHON['general']} {analysis_script}"
    for k, val in dif_analysis_cmd_dict.items():
        if val is not None:
            if isinstance(val, list):
                analysis_command += f" {k}"
                analysis_command += " " + " ".join(val)
            elif isinstance(val, bool):
                if val == True:
                    analysis_command += f" {k}"
            else:
                analysis_command += f" {k} {val}"
            print(k, val)

    print(f"Analysis command: {analysis_command}")

    if len(diffusion_outputs) < 100:
    ## Analyzing locally
        p = subprocess.Popen(analysis_command, shell=True)
        (output, err) = p.communicate()
    else:
    ## Too many structures to analyze.
    ## Running the analysis as a SLURM job.
        submit_script = "submit_diffusion_analysis.sh"
        utils.create_slurm_submit_script(filename=submit_script, name="diffusion_analysis",
                                     mem="8g", N_cores=dif_analysis_cmd_dict["--nproc"], time="0:20:00", email=EMAIL,
                                     command=analysis_command, outfile_name="output_analysis")

        p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
    return "Done"


# def analyze_diffusion_output(pdb: Annotated[str, Field(description="PDB files in script directory")], 
#                              ref: Annotated[str, Field(description="Given reference file")],
#                              params: Annotated[str, Field(description="Given weight file")],
#                              term_limit: Annotated[float, Field(description="Maximum number of allowed terms")], 
#                              SASA_limit: Annotated[float, Field(description="Highest allowed relative SASA of ligand")]) -> str:
    # SCRIPT_DIR = os.path.dirname("/ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/")  # edit this to the GitHub repo path. Throws an error by default.
    # if not os.path.exists(SCRIPT_DIR):
    #     raise RuntimeError(f"Missing SCRIPT_DIR: {SCRIPT_DIR}")
    # sys.path.append(SCRIPT_DIR+"/scripts/utils")
    # diffusion_script = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/rf_diffusion_all_atom/run_inference.py"
    # proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"

    # CONDAPATH = "/ocean/projects/cis240137p/ksubram4/anaconda3"   # edit this depending on where your Conda environments live
    # PYTHON = {"diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    #       "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    #       "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    #       "general": f"{CONDAPATH}/envs/diffusion/bin/python"}
    
    # WDIR = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/outputs"

    # if not os.path.exists(WDIR):
    #     os.makedirs(WDIR, exist_ok=True)

    # print(f"Working directory: {WDIR}")

    # USE_GPU_for_AF2 = True
    # params = [f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"]  # Rosetta params file(s)
    # LIGAND = "HBA"

    # diffusion_inputs = glob.glob(f"{SCRIPT_DIR}/input/*.pdb")
    # print(f"Found {len(diffusion_inputs)} PDB files")


    # DIFFUSION_DIR = f"{WDIR}/0_diffusion"
    # if not os.path.exists(DIFFUSION_DIR):
    #     os.makedirs(DIFFUSION_DIR, exist_ok=False)

    # os.chdir(DIFFUSION_DIR)

    # diffusion_rundirs = ['7o2g_HBA']
    # analysis_script = f"{SCRIPT_DIR}/scripts/diffusion_analysis/process_diffusion_outputs.py"

    # diffusion_outputs = []
    # for d in diffusion_rundirs:
    #     diffusion_outputs += glob.glob(f"{d}/out/*.pdb")
    
    # return pdb + " " + str(term_limit) + " " + str(SASA_limit)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')