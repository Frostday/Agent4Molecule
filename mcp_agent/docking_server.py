import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from typing import Annotated
from pydantic import Field
import subprocess

mcp = FastMCP("gromacs")

def _conda_run(argv: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run command inside the 'docking' conda env, CAPTURING output (no stdout leaks)."""
    conda = os.environ.get("CONDA_EXE", "conda")
    return subprocess.run(
        [conda, "run", "-n", "docking", *argv],  # <- no --no-capture-output
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )

def _log(workspace: str, name: str, proc: subprocess.CompletedProcess) -> None:
    """Persist child logs to a file (never print to stdout)."""
    try:
        os.makedirs(workspace, exist_ok=True)
        with open(os.path.join(workspace, f"{name}.log"), "a") as f:
            if proc.stdout:
                f.write(proc.stdout)
            if proc.stderr:
                f.write(proc.stderr)
    except Exception:
        # last resort: send to stderr, not stdout
        print(f"[warn] failed to write {name}.log", file=sys.stderr)

# @mcp.tool()
# def prepare_receptor(
#     receptor_file: Annotated[str, Field(description="Path to the receptor PDB file")],
#     output_name: Annotated[str, Field(description="Name of the output receptor PDBQT file without extension")] = "receptor_output",
#     workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
#     size_x: Annotated[float, Field(description="Size of the search box in the X dimension")] = 20.0,
#     size_y: Annotated[float, Field(description="Size of the search box in the Y dimension")] = 20.0,
#     size_z: Annotated[float, Field(description="Size of the search box in the Z dimension")] = 20.0,
#     center_x: Annotated[float, Field(description="X coordinate of the center of the search box")] = 15.590,
#     center_y: Annotated[float, Field(description="Y coordinate of the center of the search box")] = 53.903,
#     center_z: Annotated[float, Field(description="Z coordinate of the center of the search box")] = 16.917,
# ) -> str:
#     """
#     Prepares the receptor file for docking using meeko. This step create a PDBQT file of our receptor containing
#     only the polar hydrogen atoms as well as partial charges. 
#     """

#     receptor_file = os.path.join(workspace, receptor_file)
#     output_file = os.path.join(workspace, output_name + ".pdbqt")

#     cmd = [
#         "mk_prepare_receptor.py",
#         "-i", receptor_file,
#         "-o", output_name,
#         "-p",
#         "-v",
#         "--box_size", str(size_x), str(size_y), str(size_z),
#         "--box_center", str(center_x), str(center_y), str(center_z),
#     ]
    
#     proc = _conda_run(cmd, cwd=workspace)
#     _log(workspace, "prepare_receptor", proc)

#     return output_file

# @mcp.tool()
# def prepare_ligand(
#     ligand_file: Annotated[str, Field(description="Path to the ligand PDB file")],
#     output_name: Annotated[str, Field(description="Name of the output ligand PDBQT file without extension")] = "ligand_output",
#     workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
#     ) -> str:
#     """
#     Prepares the ligand file for docking using meeko. This step create a PDBQT file from a ligand molecule file
#     (preferably in the SDF format) using the Meeko python package.
#     """

#     ligand_file = os.path.join(workspace, ligand_file)
#     output_file = os.path.join(workspace, output_name + ".pdbqt")

#     cmd = [
#         "mk_prepare_ligand.py",
#         "-i", ligand_file,
#         "-o", output_file
#     ]
    
#     proc = _conda_run(cmd, cwd=workspace)
#     _log(workspace, "prepare_ligand", proc)

#     return output_file

# @mcp.tool()
# def dock_molecule(
#     receptor_file: Annotated[str, Field(description="Path to the receptor PDBQT file")],
#     ligand_file: Annotated[str, Field(description="Path to the ligand PDBQT file")],
#     output_file: Annotated[str, Field(description="Path to the output docked PDBQT file")],
#     exhaustiveness: Annotated[int, Field(description="Exhaustiveness of the search (default is 8)")] = 8,
#     workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
# ) -> str:
#     """
#     Docks a molecule using AutoDock Vina.
#     """

#     receptor_file = os.path.join(workspace, receptor_file)
#     ligand_file = os.path.join(workspace, ligand_file)
#     output_file = os.path.join(workspace, output_file)
#     config_file = os.path.join(workspace, "1iep_receptorH_output.pdbqt.box.txt")

#     # # Find box config txt file
#     # # Find the file ending in ".box.txt" in workspace
#     # ws = Path(workspace)
#     # config_file = next(ws.glob("*.box.txt"), None)
#     # if config_file is None:
#     #     raise FileNotFoundError("No .box.txt file found in workspace for docking configuration.")
    
#     vina_command = [
#         "vina",
#         "--receptor", receptor_file,
#         "--ligand", ligand_file,
#         "--config", config_file,
#     ]

#     proc = _conda_run(vina_command, cwd=workspace)
#     _log(workspace, "vina", proc)

#     return "Successfully finished task."

@mcp.tool()
def dock_pipeline(
    receptor_file: Annotated[str, Field(description="Path to the receptor PDB file")],
    ligand_file: Annotated[str, Field(description="Path to the ligand PDB file")],
    prepared_receptor_name: Annotated[str, Field(description="Name of the output receptor PDBQT file without extension")] = "receptor_output",
    prepared_ligand_name: Annotated[str, Field(description="Name of the output ligand PDBQT file without extension")] = "ligand_output",
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
    size_x: Annotated[float, Field(description="Size of the search box in the X dimension")] = 20.0,
    size_y: Annotated[float, Field(description="Size of the search box in the Y dimension")] = 20.0,
    size_z: Annotated[float, Field(description="Size of the search box in the Z dimension")] = 20.0,
    center_x: Annotated[float, Field(description="X coordinate of the center of the search box")] = 15.590,
    center_y: Annotated[float, Field(description="Y coordinate of the center of the search box")] = 53.903,
    center_z: Annotated[float, Field(description="Z coordinate of the center of the search box")] = 16.917,
    output_file: Annotated[str, Field(description="Path to the output docked PDBQT file")] = "docked.pdbqt",
    exhaustiveness: Annotated[int, Field(description="Exhaustiveness of the search (default is 8)")] = 8,
) -> str:
    """
    Docking: prepare receptor -> prepare ligand -> dock molecule
    """

    # prepare receptor
    receptor_path = os.path.join(workspace, receptor_file)
    prepared_receptor_path = os.path.join(workspace, prepared_receptor_name + ".pdbqt")

    receptor_cmd = [
        "mk_prepare_receptor.py",
        "-i", receptor_path,
        "-o", prepared_receptor_name,
        "-p",
        "-v",
        "--box_size", str(size_x), str(size_y), str(size_z),
        "--box_center", str(center_x), str(center_y), str(center_z),
    ]

    _conda_run(receptor_cmd, cwd=workspace)

    # prepare ligand
    ligand_path = os.path.join(workspace, ligand_file)
    prepared_ligand_path = os.path.join(workspace, prepared_ligand_name + ".pdbqt")

    ligand_cmd = [
        "mk_prepare_ligand.py",
        "-i", ligand_path,
        "-o", prepared_ligand_path
    ]

    _conda_run(ligand_cmd, cwd=workspace)

    # dock molecule

    # pick/verify config
    cfg = os.path.join(workspace, Path(prepared_receptor_path).stem + ".box.txt")
    if not os.path.exists(cfg):
        # fallback: auto-pick if exactly one; otherwise return an error string
        matches = list(Path(workspace).glob("*.box.txt"))
        if len(matches) != 1:
            return {"error": "Config .box.txt not found or ambiguous", "candidates": [m.name for m in matches]}
        cfg = str(matches[0])

    # run vina
    vina_cmd = [
        "vina",
        "--receptor", prepared_receptor_path,
        "--ligand", prepared_ligand_path,
        "--config", cfg,
        "--out", os.path.join(workspace, output_file),
        "--exhaustiveness", str(exhaustiveness),
    ]

    _conda_run(vina_cmd, cwd=workspace)

    return "Successfully finished task."


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')