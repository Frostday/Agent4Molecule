import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from typing import Annotated
from pydantic import Field
import subprocess
import os

mcp = FastMCP("gromacs")

def _conda_run(argv: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run command inside the 'docking' conda env, CAPTURING output (no stdout leaks)."""
    conda = os.environ.get("CONDA_EXE", "conda")
    return subprocess.run(
        [conda, "run", "-n", "vina", *argv],  # <- no --no-capture-output
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )

@mcp.tool()
def smiles_to_pdb(
    smiles: Annotated[str, Field(description="SMILES string of the molecule to convert")],
    output_file: Annotated[str, Field(description="Path to the output PDB file")] = "smiles_output.pdb",
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
) -> str:
    """
    Converts a SMILES string to a PDB file using Open Babel.
    """

    output_path = os.path.join(workspace, output_file)

    cmd = [
        "obabel",
        f"-:{smiles}",
        "-O", output_path,
        "--gen3d"
    ]

    _conda_run(cmd, cwd=workspace)

    return output_path

@mcp.tool()
def sdf_to_pdb(
    sdf_file: Annotated[str, Field(description="Path to the input SDF file")] = "substrate.sdf",
    output_file: Annotated[str, Field(description="Path to the output PDB file")] = "sdf_output.pdb",
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
) -> str:
    """
    Converts an SDF file to a PDB file using Open Babel.
    """

    sdf_path = os.path.join(workspace, sdf_file)
    output_path = os.path.join(workspace, output_file)

    cmd = [
        "obabel",
        sdf_path,
        "-O", output_path,
        "--gen3d"
    ]

    _conda_run(cmd, cwd=workspace)

    return output_path

@mcp.tool()
def mol_to_pdb(
    mol_file: Annotated[str, Field(description="Path to the input MOL file")] = "substrate.mol",
    output_file: Annotated[str, Field(description="Path to the output PDB file")] = "mol_output.pdb",
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
) -> str:
    """
    Converts a MOL file to a PDB file using Open Babel.
    """

    mol_path = os.path.join(workspace, mol_file)
    output_path = os.path.join(workspace, output_file)

    cmd = [
        "obabel",
        mol_path,
        "-O", output_path,
        "--gen3d"
    ]

    _conda_run(cmd, cwd=workspace)

    return output_path

@mcp.tool()
def smiles_to_sdf(
    smiles: Annotated[str, Field(description="SMILES string of the molecule to convert")],
    output_file: Annotated[str, Field(description="Path to the output SDF file")] = "smiles_output.sdf",
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
) -> str:
    """
    Converts a SMILES string to an SDF file using RDKit.
    """

    output_path = os.path.join(workspace, output_file)

    cmd = [
        "python", "/jet/home/eshen3/Agent4Molecule/mcp_agent/util/to_sdf.py",
        "--smiles", smiles,
        "--out", output_path,
    ]

    _conda_run(cmd, cwd=workspace)
    
    return output_path

@mcp.tool()
def mol_to_sdf(
    mol_file: Annotated[str, Field(description="Path to the input MOL file")] = "substrate.mol",
    output_file: Annotated[str, Field(description="Path to the output SDF file")] = "mol_output.sdf",
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
) -> str:
    """
    Converts a MOL file to an SDF file using Open Babel.
    """

    mol_path = os.path.join(workspace, mol_file)
    output_path = os.path.join(workspace, output_file)

    cmd = [
        "python", "/jet/home/eshen3/Agent4Molecule/mcp_agent/util/to_sdf.py",
        "--infile", mol_path,
        "--out", output_path,
        "--no-gen3d"
    ]

    _conda_run(cmd, cwd=workspace)

    return output_path

@mcp.tool()
def pdb_to_sdf(
    pdb_file: Annotated[str, Field(description="Path to the input PDB file")] = "substrate.pdb",
    output_file: Annotated[str, Field(description="Path to the output SDF file")] = "pdb_output.sdf",
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
) -> str:
    """
    Converts a PDB file to an SDF file using Open Babel.
    """

    pdb_path = os.path.join(workspace, pdb_file)
    tmp_sdf_path = os.path.join(workspace, "tmp_" + output_file)
    output_path = os.path.join(workspace, output_file)

    # use obabel to convert pdb to sdf
    obabel_cmd = [
        "obabel",
        pdb_path,
        "-O", tmp_sdf_path,
        "-h"
    ]

    _conda_run(obabel_cmd, cwd=workspace)

    # clean single fragment
    clean_cmd = [
        "python", "/jet/home/eshen3/Agent4Molecule/mcp_agent/util/clean_fragment.py",
        tmp_sdf_path, output_path
    ]

    _conda_run(clean_cmd, cwd=workspace)

    # remove temporary file
    os.remove(tmp_sdf_path)

    return output_path


@mcp.tool()
def dock_pipeline(
    receptor_file: Annotated[str, Field(description="Path to the receptor PDB file")],
    ligand_file: Annotated[str, Field(description="Path to the ligand SDF file")],
    size_x: Annotated[float, Field(description="Size of the search box in the X dimension")] = 80.0,
    size_y: Annotated[float, Field(description="Size of the search box in the Y dimension")] = 80.0,
    size_z: Annotated[float, Field(description="Size of the search box in the Z dimension")] = 80.0,
    center_x: Annotated[float, Field(description="X coordinate of the center of the search box")] = 0.0,
    center_y: Annotated[float, Field(description="Y coordinate of the center of the search box")] = 0.0,
    center_z: Annotated[float, Field(description="Z coordinate of the center of the search box")] = 0.0,
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
    prepared_receptor_name: Annotated[str, Field(description="Name of the output receptor PDBQT file without extension")] = "receptor_output",
    prepared_ligand_name: Annotated[str, Field(description="Name of the output ligand PDBQT file without extension")] = "ligand_output",
    output_file: Annotated[str, Field(description="Path to the output docked PDBQT file")] = "docked.pdbqt",
    exhaustiveness: Annotated[int, Field(description="Exhaustiveness of the search (default is 8)")] = 8,
) -> str:
    """
    Docking: prepare receptor -> prepare ligand -> dock molecule
    """
    os.makedirs(workspace, exist_ok=True)

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

    with open(os.path.join(workspace, output_file), "r") as f:
        content = f.read()
        content = re.findall(r"REMARK VINA RESULT:\s+([-+]?\d*\.?\d+)", content)
    binding_affinity = content[0]

    return f"Successfully finished task. Protein pdbqt file location: {os.path.join(workspace, prepared_receptor_name + ".pdbqt")}\nDocked ligand pdbqt file location: {os.path.join(workspace, output_file)}\nBinding Affinity: {binding_affinity}"


if __name__ == "__main__":
    mcp.run(transport='stdio')
