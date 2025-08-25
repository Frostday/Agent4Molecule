import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
from mcp.server.fastmcp import FastMCP
from typing import Annotated
import os,glob
from pydantic import Field
import subprocess
import time


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
        #SBATCH -N 1
        #SBATCH -p GPU-shared
        #SBATCH -t 24:00:00
        #SBATCH --gres=gpu:1
        #SBATCH --job-name=gmx_copilot
        #SBATCH --output={log_file}
        #SBATCH --error={err_file}

        source ~/.bashrc
        conda activate gromacs_env
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
    workspace: Annotated[str, Field(description="Workspace containing GROMACS Copilot outputs")]= "/ocean/projects/cis240137p/eshen3/gromacs_copilot/md_workspace"
    ) -> str:
    """
    Automatically finds the latest GROMACS Copilot output and visualizes it.
    """
    # Plot RMSD
    rmsd_file = os.path.join(workspace, "analysis/rmsd.xvg")
    if os.path.exists(rmsd_file):
        data = []
        with open(rmsd_file, "r") as file:
            for line in file:
                if line.startswith(("@", "#")):
                    continue
                parts = re.split(r'\s+', line.strip())
                if len(parts) == 2:
                    time_val, rmsd_val = map(float, parts)
                    data.append((time_val, rmsd_val))
        
        df = pd.DataFrame(data, columns=["Time (ns)", "RMSD (nm)"])
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="Time (ns)", y="RMSD (nm)", linewidth=2)
        plt.title("RMSD of Protein Over Time")
        plt.xlabel("Time (ns)")
        plt.ylabel("RMSD (nm)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(workspace, "rmsd_plot.png"))
        plt.close()


    # Plot RMSF
    rmsf_file = os.path.join(workspace, "analysis/rmsf.xvg")
    if os.path.exists(rmsf_file):
        rmsf_data = []
        with open(rmsf_file, "r") as file:
            for line in file:
                if line.startswith(("@", "#")):
                    continue
                parts = re.split(r'\s+', line.strip())
                if len(parts) == 2:
                    residue, rmsf_val = map(float, parts)
                    rmsf_data.append((residue, rmsf_val))
        
        rmsf_df = pd.DataFrame(rmsf_data, columns=["Residue", "RMSF (nm)"])
        plt.figure(figsize=(12, 5))
        sns.lineplot(x='Residue', y='RMSF (nm)', data=rmsf_df, marker='o')
        plt.title('RMSF per Residue')
        plt.xlabel('Residue Number')
        plt.ylabel('RMSF (nm)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(workspace, "rmsf_plot.png"))
        plt.close()
    
    # Find pdb file name
    pdb_files = glob.glob(os.path.join(workspace, "*.pdb"))
    return f"Visualization complete for {os.path.basename(pdb_files[0])}. RMSD and RMSF plots saved to workspace if available."