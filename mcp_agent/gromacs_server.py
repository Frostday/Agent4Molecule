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


@mcp.tool()
def run_gromacs_copilot(
    workspace: Annotated[str, Field(description="Working directory for MD simulations")] = "/ocean/projects/cis240137p/eshen3/gromacs_copilot/md_workspace",
    prompt: Annotated[str, Field(description="Natural language prompt to control GROMACS Copilot")],
    api_key: Annotated[str, Field(description="API key for LLM service")],
    model: Annotated[str, Field(description="LLM model name, e.g., gpt-4o, deepseek-chat, gemini-2.0-flash")],
    api_url: Annotated[str, Field(description="URL for LLM API")],
    mode: Annotated[str, Field(description="Copilot mode: copilot, agent, or debug")] = "agent"
) -> str:

    cmd = [
        "gmx_copilot",
        "--workspace", workspace,
        "--prompt", prompt,
        "--api-key", api_key,
        "--model", model,
        "--url", api_url,
        "--mode", mode
    ]

    print(f"Running GROMACS Copilot with: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"GROMACS Copilot failed:\n{result.stderr}")

    return result.stdout


@mcp.tool()
def visualize_latest_gromacs_output(
    workspace: Annotated[str, Field(description="Workspace containing GROMACS Copilot outputs")] = "/ocean/projects/cis240137p/eshen3/gromacs_copilot/md_workspace"
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

def main():
    # Start the MCP server with stdio transport
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()