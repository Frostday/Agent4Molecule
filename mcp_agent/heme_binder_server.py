import os
import glob
from typing import Annotated
from pydantic import Field
import json
import subprocess
import time

from run_utils import extract_job_id, RF_DIFFUSION_CONFIG

HEME_BINDER_PATH = "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/"
CONDAPATH = "/ocean/projects/cis240137p/dgarg2/miniconda3"
diffusion_script = os.path.join("/ocean/projects/cis240137p/dgarg2/github/rf_diffusion_all_atom/run_inference.py")
import sys
sys.path.append(HEME_BINDER_PATH)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("heme_binder")

# SETUP
WDIR = os.path.join(HEME_BINDER_PATH, "agent_output/")
SCRIPT_DIR = os.path.dirname(HEME_BINDER_PATH)
assert os.path.exists(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR+"/scripts/utils")
import utils
proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"
AF2_script = f"{SCRIPT_DIR}/scripts/af2/af2.py"
PYTHON = {
    "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    "general": f"{CONDAPATH}/envs/diffusion/bin/python"
}
if not os.path.exists(WDIR):
    os.makedirs(WDIR, exist_ok=True)
USE_GPU_for_AF2 = True

# INPUTS
# params = [f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"]
# LIGAND = "HBA"
# diffusion_inputs = ['/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb']


@mcp.tool()
def run_rf_diffusion(
    input_pdb: Annotated[str, Field(description="Input PDB file")],
    ligand: Annotated[str, Field(description="Ligand name")],
    N_designs: Annotated[int, Field(description="Number of designs to generate")] = 5,
    T_steps: Annotated[int, Field(description="Number of diffusion steps")] = 200,
) -> str:
    DIFFUSION_DIR = f"{WDIR}/0_diffusion"
    if not os.path.exists(DIFFUSION_DIR):
        os.makedirs(DIFFUSION_DIR, exist_ok=False)
    os.chdir(DIFFUSION_DIR)
    config = RF_DIFFUSION_CONFIG.format(
        T_steps=T_steps,
        N_designs=N_designs,
        LIGAND=ligand
    )
    # estimated_time = 3.5 * T_steps * N_designs
    # print(f"Estimated time to produce {N_designs} designs = {estimated_time/60:.0f} minutes")
    with open("config.yaml", "w") as file:
        file.write(config)
    # print(f"Wrote config file to {os.path.realpath('config.yaml')}")

    diffusion_inputs = [input_pdb]
    commands_diffusion = []
    cmds_filename = "commands_diffusion"
    diffusion_rundirs = []
    with open(cmds_filename, "w") as file:
        for p in diffusion_inputs:
            pdbname = os.path.basename(p).replace(".pdb", "")
            os.makedirs(pdbname, exist_ok=True)
            cmd = f"cd {pdbname} ; {PYTHON['diffusion']} {diffusion_script} --config-dir=../ "\
                f"--config-name=config.yaml inference.input_pdb={p} "\
                f"inference.output_prefix='./out/{pdbname}_dif' > output.log ; cd ..\n"
            commands_diffusion.append(cmd)
            diffusion_rundirs.append(pdbname)
            file.write(cmd)
    # print(f"An example diffusion command that was generated:\n   {cmd}")

    submit_script = "submit_diffusion.sh"
    utils.create_slurm_submit_script(
        filename=submit_script, 
        # gpu=True, 
        # gres="v100-32:1",
        # mem="8g", 
        N_cores=2, 
        time="1:00:00", 
        array=len(commands_diffusion), 
        array_commandfile=cmds_filename
    )
    # print(f"Writing diffusion submission script to {submit_script}")
    # print(f"{len(commands_diffusion)} diffusion jobs to run")

    p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("Heme Binder RFDiffusion Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(300)

    output_preds = []
    rfa_logs = []
    with open(os.path.join(DIFFUSION_DIR, "output.log"), "r") as f:
        logs = f.read()
    with open(os.path.join(DIFFUSION_DIR, "output.err"), "r") as f:
        errors = f.read()
    for rundir in diffusion_rundirs:
        diffusion_outputs = glob.glob(f"{rundir}/out/*.pdb")
        for out in diffusion_outputs:
            with open(out, "r") as f:
                output_preds.append(f"File: {os.path.join(DIFFUSION_DIR, out)}\n\n" + f.read())
        with open(os.path.join(rundir, "output.log"), "r") as f:
            rfa_logs.append(f"File: {os.path.join(DIFFUSION_DIR, rundir, 'output.log')}\n\n" + f.read())
            
    return "Predicted structure(s) from RFDiffusionAA:\n\n----------\n" + "\n----------\n----------\n".join(output_preds) + "\n----------\n\n" + "Log File:\n\n----------\n" + logs + "\n----------\n\n" + "RFDiffusionAA Log Files:\n\n----------\n" + "\n----------\n----------\n".join(rfa_logs) + "\n----------\n\n" + "Error File:\n\n----------\n" + errors


@mcp.tool()
def rf_diffusion_analysis() -> str:
    return


def cleanup():
    os.system(f"rm -f /ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/output.txt")
    os.system(f"rm -rf {WDIR}/*")


if __name__ == "__main__":
    # mcp.run(transport='stdio')
    cleanup()
    # output = run_rf_diffusion(input_pdb="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb", ligand="HBA", N_designs=2, T_steps=200)
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/output.txt", "w") as f:
    #     f.write(output)

