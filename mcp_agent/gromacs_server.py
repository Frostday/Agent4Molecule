from mcp.server.fastmcp import FastMCP
from typing import Any,Annotated,Optional
import httpx
import os,sys,glob
from pydantic import Field
import utils,json
from google import genai
import getpass
import subprocess
import time
import importlib
from shutil import copy2
from google.genai.types import Content, Part


mcp = FastMCP("gromacs")

def extract_job_id(output: str) -> str:
    """Extracts the job ID from the output of the sbatch command."""
    lines = output.split('\n')
    for line in lines:
        if "Submitted batch job" in line:
            return line.split()[-1]
    return ""


@mcp.tool()
def run_gromacs_copilot(
    prompt: Annotated[str, Field(description="Natural language prompt to control GROMACS Copilot")],
    api_key: Annotated[str, Field(description="API key for LLM service")],
    model: Annotated[str, Field(description="LLM model name, e.g., gpt-4o, deepseek-chat, gemini-2.0-flash")],
    api_url: Annotated[str, Field(description="URL for LLM API")],
    workspace: Annotated[str, Field(description="Working directory for MD simulations")] = "/ocean/projects/cis240137p/eshen3/gromacs_copilot/md_workspace",
    mode: Annotated[str, Field(description="Copilot mode: copilot, agent, or debug")] = "agent"
    ) -> str:
    """
    Submits a SLURM job to run GROMACS Copilot and waits for completion.
    """
    slurm_script = os.path.join(workspace, "run_copilot.slurm")
    log_file = os.path.join(workspace, "copilot_output.log")
    err_file = os.path.join(workspace, "copilot_output.err")

    cmd = f"gmx_copilot --workspace {workspace} --prompt \"{prompt}\" --api-key {api_key} --model {model} --url {api_url} --mode {mode}"

    with open(slurm_script, "w") as f:
        f.write(f"""#!/bin/bash
        #SBATCH --job-name=gmx_copilot
        #SBATCH --output={log_file}
        #SBATCH --error={err_file}
        #SBATCH --time=01:00:00
        #SBATCH --partition=RM
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem=8G


        source ~/.bashrc
        conda activate mcp-agent
        {cmd}
        """)

    p = subprocess.Popen(["sbatch", slurm_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()

    if p.returncode != 0:
        raise RuntimeError(f"Failed to submit SLURM job:\n{err.decode()}")

    output_str = output.decode("utf-8")
    print(output_str)
    job_id = output_str.strip().split()[-1]


    # Wait for job to complete
    print(f"Submitted job {job_id}. Waiting for it to complete...")
    while True:
        q = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        qout, _ = q.communicate()
        if job_id not in qout.decode("utf-8"):
            break
        print("Job still running...")
        time.sleep(60)


    # Collect and return outputs
    with open(log_file, "r") as f:
        logs = f.read()
    with open(err_file, "r") as f:
        errors = f.read()


    return f"Job {job_id} completed.\n\nLog Output:\n{logs}\n\nErrors:\n{errors}"

@mcp.tool()
def visualize_latest_gromacs_output(
    workspace: Annotated[str, Field(description="Workspace containing GROMACS Copilot outputs")]
    ) -> str:
    """
    Automatically finds the latest GROMACS Copilot output and visualizes it in PyMOL.
    """
    try:
        import pymol2
    except ImportError:
        raise RuntimeError("PyMOL2 library is not installed. Please install it with 'pip install pymol-open-source' or use your PyMOL installation.")


    # Look for .gro and .xtc files in the workspace
    pdb_files = sorted(glob.glob(os.path.join(workspace, "*.gro")) + glob.glob(os.path.join(workspace, "*.pdb")), reverse=True)
    traj_files = sorted(glob.glob(os.path.join(workspace, "*.xtc")) + glob.glob(os.path.join(workspace, "*.trr")), reverse=True)


    if not pdb_files:
        raise FileNotFoundError("No structure (.gro or .pdb) file found in the workspace.")


    pdb_file = pdb_files[0]
    trajectory_file = traj_files[0] if traj_files else None


    with pymol2.PyMOL() as pymol:
        pymol.cmd.load(pdb_file, "protein")
    if trajectory_file:
        pymol.cmd.load_traj(trajectory_file, "protein")
    pymol.cmd.show("cartoon", "protein")
    pymol.cmd.color("cyan", "protein")
    pymol.cmd.orient("protein")


    return f"Visualization complete for {os.path.basename(pdb_file)}."