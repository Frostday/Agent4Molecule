from mcp.server.fastmcp import FastMCP
import subprocess

mcp = FastMCP("gromacs")

def extract_job_id(output: str) -> str:
    """Extracts the job ID from the output of the sbatch command."""
    lines = output.split('\n')
    for line in lines:
        if "Submitted batch job" in line:
            return line.split()[-1]
    return ""

@mcp.tool()
def prepare_receptor(
    receptor_file: Annotates[str, Field(description="Path to the receptor PDB file")],
    output_file: Annotates[str, Field(description="Path to the output PDBQT file")],
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
    size_x: Annotates[float, Field(description="Size of the search box in the X dimension")] = 20.0,
    size_y: Annotates[float, Field(description="Size of the search box in the Y dimension")] = 20.0,
    size_z: Annotates[float, Field(description="Size of the search box in the Z dimension")] = 20.0,
    center_x: Annotates[float, Field(description="X coordinate of the center of the search box")] = 0.0,
    center_y: Annotates[float, Field(description="Y coordinate of the center of the search box")] = 0.0,
    center_z: Annotates[float, Field(description="Z coordinate of the center of the search box")] = 0.0,
) -> str:
    """
    Prepares the receptor file for docking using meeko.
    
    Parameters:
        receptor_file (str): Path to the input receptor PDB file.
        output_file (str): Path to save the prepared receptor PDBQT file.
        size_x (float): Size of the search box in the X dimension.
        size_y (float): Size of the search box in the Y dimension.
        size_z (float): Size of the search box in the Z dimension.
        center_x (float): X coordinate of the center of the search box.
        center_y (float): Y coordinate of the center of the search box.
        center_z (float): Z coordinate of the center of the search box.

    
    Returns:
        str: Path to the output prepared receptor PDBQT file.
    """

    receptor_file = os.path.join(workspace, receptor_file)
    output_file = os.path.join(workspace, output_file)

    command = [
        "mk_prepare_receptor.py",
        "-i", receptor_file,
        "-o", output_file
        "-p",
        "-v",
        "--box_size", f"{size_x},{size_y},{size_z}",
        "--box_center", f"{center_x},{center_y},{center_z}"
    ]
    
    subprocess.run(command, check=True)
    return output_file

def prepare_ligand(
    ligand_file: Annotated[str, Field(description="Path to the ligand PDB file")],
    output_file: Annotated[str, Field(description="Path to the output PDBQT file")],
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
    ) -> str:
    """
    Prepares the ligand file for docking using AutoDockTools.
    
    Parameters:
        ligand_file (str): Path to the input ligand PDB file.
        output_file (str): Path to save the prepared ligand PDBQT file.
    
    Returns:
        str: Path to the output prepared ligand PDBQT file.
    """

    ligand_file = os.path.join(workspace, ligand_file)
    output_file = os.path.join(workspace, output_file)

    command = [
        "prepare_ligand4.py",
        "-l", ligand_file,
        "-o", output_file
    ]
    
    subprocess.run(command, check=True)
    return output_file

@mcp.tool()
def dock_molecule(
    receptor_file: Annotated[str, Field(description="Path to the receptor PDBQT file")],
    ligand_file: Annotated[str, Field(description="Path to the ligand PDBQT file")],
    output_file: Annotated[str, Field(description="Path to the output docked PDBQT file")],
    exhaustiveness: Annotated[int, Field(description="Exhaustiveness of the search (default is 8)")] = 8,
    workspace: Annotated[str, Field(description="Working directory for docking files")] = "/ocean/projects/cis240137p/eshen3/docking",
) -> str:
    """
    Docks a molecule using AutoDock Vina.
    
    Parameters:
        receptor_file (str): Path to the receptor PDBQT file.
        ligand_file (str): Path to the ligand PDBQT file.
        output_file (str): Path to save the docked output PDBQT file.
        exhaustiveness (int): Exhaustiveness of the search (default is 8).
    
    Returns:
        str: Path to the output docked PDBQT file.
    """

    receptor_file = os.path.join(workspace, receptor_file)
    ligand_file = os.path.join(workspace, ligand_file)
    output_file = os.path.join(workspace, output_file)

    # Find box config txt file
    # The file name should be receptor name without "pdbqt" + ".box.txt"
    config_file = receptor_file.replace(".pdbqt", ".box.txt")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")
    
    vina_command = [
        "vina",
        "--receptor", receptor_file,
        "--ligand", ligand_file,
        "--config", config_file
        "--exhaustiveness", str(exhaustiveness),
        "--out", output_file,
    ]

    subprocess.run(vina_command, check=True)
    return output_file

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')