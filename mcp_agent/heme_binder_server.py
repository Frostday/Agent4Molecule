import os
import glob
from typing import Annotated
from pydantic import Field
import json
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
from shutil import copy2
import re

from run_utils import extract_job_id, RF_DIFFUSION_CONFIG, cst_template

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
    "general": f"{CONDAPATH}/envs/diffusion/bin/python",
    "vina": f"{CONDAPATH}/envs/vina/bin/python"
}
if not os.path.exists(WDIR):
    os.makedirs(WDIR, exist_ok=True)
PARAMS_GENERATION = "/ocean/projects/cis240137p/dgarg2/github/pyrosetta/rosetta.source.release-408/main/source/scripts/python/public/molfile_to_params.py"
combine_protein_ligand_file = "/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/util/combine_protein_ligand.py"
separate_protein_ligand_file = "/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/util/sep_protein_ligand.py"


@mcp.tool()
def separate_ligand_protein_pdb(
    pdb_file: Annotated[str, Field(description="Input pdb file (protein-ligand complex)")]
):
    INPUT_DIR = f"{WDIR}/start"
    os.system(f"rm -rf {INPUT_DIR}/*")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.system(f"cp {pdb_file} {INPUT_DIR}/protein_ligand.pdb")
    os.chdir(INPUT_DIR)

    command = f"{PYTHON['general']} {separate_protein_ligand_file} {INPUT_DIR}/protein_ligand.pdb {INPUT_DIR}/protein.pdb {INPUT_DIR}/ligand.pdb"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    
    command = f"obabel -ipdb ligand.pdb -osdf -O ligand.sdf"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    # clean_cmd = f"{PYTHON['vina']} /ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/util/clean_fragment.py {INPUT_DIR}/ligand.sdf {INPUT_DIR}/ligand_cleaned.sdf" 
    # p = subprocess.Popen(clean_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # (output, err) = p.communicate()

    # return f"Protein pdb file generated at: {INPUT_DIR}/protein.pdb\nLigand pdb file generated at: {INPUT_DIR}/ligand.pdb\nLigand sdf file generated at: {INPUT_DIR}/ligand.sdf"
    return f"Protein pdb file generated at: {INPUT_DIR}/protein.pdb\nLigand sdf file generated at: {INPUT_DIR}/ligand.sdf"


@mcp.tool()
def convert_ligand_pdb_to_sdf(
    ligand_pdb_file: Annotated[str, Field(description="Input pdb file (ligand)")]
):
    INPUT_DIR = f"{WDIR}/start"
    os.system(f"rm -rf {INPUT_DIR}/*")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.system(f"cp {ligand_pdb_file} {INPUT_DIR}/ligand.pdb")
    os.chdir(INPUT_DIR)

    command = f"obabel -ipdb ligand.pdb -osdf -O ligand.sdf"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    return f"Ligand sdf file generated at: {INPUT_DIR}/ligand.sdf"


@mcp.tool()
def generate_cst_file(
    res1_atoms: Annotated[list[str], Field(description="Ligand atoms (like \"FE1\", \"N4\", \"C19\")")],
    res1_residue: Annotated[str, Field(description="Ligand name (like \"HBA\")")],
    res2_atoms: Annotated[list[str], Field(description="Protein atoms (like \"SH1\")")],
    res2_residue: Annotated[str, Field(description="Protein residue (like \"CYS\")")],
    distance_mean: Annotated[float, Field(description="Distance mean")] = 2.5,
    distance_tol: Annotated[float, Field(description="Distance tolerance")] = 0.15,
    distance_weight: Annotated[float, Field(description="Distance weight")] = 100,
    distance_idxA: Annotated[int, Field(description="Distance index A")] = 1,
    distance_idxB: Annotated[int, Field(description="Distance index B")] = 1,
    angleA_mean: Annotated[float, Field(description="Angle A mean")] = 87.5,
    angleA_tol: Annotated[float, Field(description="Angle A tolerance")] = 5,
    angleA_weight: Annotated[float, Field(description="Angle A weight")] = 100,
    angleA_periodicity: Annotated[int, Field(description="Angle A periodicity")] = 360,
    angleA_idx: Annotated[int, Field(description="Angle A index")] = 1,
    angleB_mean: Annotated[float, Field(description="Angle B mean")] = 108.9,
    angleB_tol: Annotated[float, Field(description="Angle B tolerance")] = 5,
    angleB_weight: Annotated[float, Field(description="Angle B weight")] = 75,
    angleB_periodicity: Annotated[int, Field(description="Angle B periodicity")] = 360,
    angleB_idx: Annotated[int, Field(description="Angle B index")] = 1,
    torsA_mean: Annotated[float, Field(description="Torsion A mean")] = 87.2,
    torsA_tol: Annotated[float, Field(description="Torsion A tolerance")] = 5,
    torsA_weight: Annotated[float, Field(description="Torsion A weight")] = 75,
    torsA_periodicity: Annotated[int, Field(description="Torsion A periodicity")] = 360,
    torsA_idx: Annotated[int, Field(description="Torsion A index")] = 1,
    torsAB_mean: Annotated[float, Field(description="Torsion AB mean")] = 86.7,
    torsAB_tol: Annotated[float, Field(description="Torsion AB tolerance")] = 15,
    torsAB_weight: Annotated[float, Field(description="Torsion AB weight")] = 0,
    torsAB_periodicity: Annotated[int, Field(description="Torsion AB periodicity")] = 90,
    torsAB_idx: Annotated[int, Field(description="Torsion AB index")] = 2,
    torsB_mean: Annotated[float, Field(description="Torsion B mean")] = 108.5,
    torsB_tol: Annotated[float, Field(description="Torsion B tolerance")] = 20,
    torsB_weight: Annotated[float, Field(description="Torsion B weight")] = 25,
    torsB_periodicity: Annotated[int, Field(description="Torsion B periodicity")] = 360,
    torsB_idx: Annotated[int, Field(description="Torsion B index")] = 2,
    max_dunbrack: Annotated[float, Field(description="Max Dunbrack energy")] = 5
) -> str:
    INPUT_DIR = f"{WDIR}/start"
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.chdir(INPUT_DIR)

    with open("cst_file.cst", "w") as file:
        file.write(cst_template.format(
            res1_atoms=" ".join(res1_atoms),
            res1_residue=res1_residue,
            res2_atoms=" ".join(res2_atoms),
            res2_residue=res2_residue,
            distance_mean=distance_mean,
            distance_tol=distance_tol,
            distance_weight=distance_weight,
            distance_idxA=distance_idxA,
            distance_idxB=distance_idxB,
            angleA_mean=angleA_mean,
            angleA_tol=angleA_tol,
            angleA_weight=angleA_weight,
            angleA_periodicity=angleA_periodicity,
            angleA_idx=angleA_idx,
            angleB_mean=angleB_mean,
            angleB_tol=angleB_tol,
            angleB_weight=angleB_weight,
            angleB_periodicity=angleB_periodicity,
            angleB_idx=angleB_idx,
            torsA_mean=torsA_mean,
            torsA_tol=torsA_tol,
            torsA_weight=torsA_weight,
            torsA_periodicity=torsA_periodicity,
            torsA_idx=torsA_idx,
            torsAB_mean=torsAB_mean,
            torsAB_tol=torsAB_tol,
            torsAB_weight=torsAB_weight,
            torsAB_periodicity=torsAB_periodicity,
            torsAB_idx=torsAB_idx,
            torsB_mean=torsB_mean,
            torsB_tol=torsB_tol,
            torsB_weight=torsB_weight,
            torsB_periodicity=torsB_periodicity,
            torsB_idx=torsB_idx,
            max_dunbrack=max_dunbrack
        ))
    
    return f"CST file generated at: {INPUT_DIR}/cst_file.cst"


@mcp.tool()
def generate_ligand_params_file(
    ligand_sdf_file: Annotated[str, Field(description="Input sdf file (ligand): use separate function if not present")],
    ligand_pdb_file: Annotated[str, Field(description="Input pdb file (ligand): use the pdb file from the user prompt")],
    ligand: Annotated[str, Field(description="Ligand name")],
):
    INPUT_DIR = f"{WDIR}/inputs"
    os.system(f"rm -rf {INPUT_DIR}/*")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.system(f"cp {ligand_sdf_file} {INPUT_DIR}/ligand.sdf")
    os.chdir(INPUT_DIR)

    command = f"obabel {INPUT_DIR}/ligand.sdf -O {INPUT_DIR}/ligand_withH.sdf -h"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    command = f"{PYTHON['diffusion']} {PARAMS_GENERATION} -n {ligand} -p {ligand} --conformers-in-one-file {INPUT_DIR}/ligand_withH.sdf"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    os.system(f"cp {glob.glob(f'{INPUT_DIR}/*.params')[-1]} {INPUT_DIR}/{ligand}.params")

    os.system(f"cp {ligand_pdb_file} {INPUT_DIR}/{ligand}.pdb")
    os.system(f"cp {INPUT_DIR}/{ligand}.pdb {INPUT_DIR}/{ligand}_conformers.pdb")

    return f"Ligand params file generated at: {INPUT_DIR}/{ligand}.params"


# @mcp.tool()
# def convert_ligand_pdb_to_sdf_for_docking(
#     pdb_path: Annotated[str, Field(description="Path to the input PDB file")]
# ) -> str:
#     INPUT_DIR = f"{WDIR}/inputs"
#     os.makedirs(INPUT_DIR, exist_ok=True)
#     os.system(f"cp {pdb_path} {INPUT_DIR}/ligand.pdb")
#     os.chdir(INPUT_DIR)
    
#     obabel_cmd = f"obabel {pdb_path} -O {INPUT_DIR}/ligand_sdf.sdf -h"
#     p = subprocess.Popen(obabel_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     (output, err) = p.communicate()

#     clean_cmd = f"{PYTHON['vina']} /ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/util/clean_fragment.py {INPUT_DIR}/ligand_sdf.sdf {INPUT_DIR}/ligand_cleaned.sdf" 
#     p = subprocess.Popen(clean_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     (output, err) = p.communicate()

#     return f"Ligand sdf file generated at: {INPUT_DIR}/ligand_cleaned.sdf"


# @mcp.tool()
# def run_docking_pipeline(
#     receptor_path: Annotated[str, Field(description="Path to the receptor PDB file")],
#     ligand_path: Annotated[str, Field(description="Path to the ligand SDF file")],
#     size_x: Annotated[float, Field(description="Size of the search box in the X dimension")],
#     size_y: Annotated[float, Field(description="Size of the search box in the Y dimension")],
#     size_z: Annotated[float, Field(description="Size of the search box in the Z dimension")],
#     center_x: Annotated[float, Field(description="X coordinate of the center of the search box")],
#     center_y: Annotated[float, Field(description="Y coordinate of the center of the search box")],
#     center_z: Annotated[float, Field(description="Z coordinate of the center of the search box")],
#     exhaustiveness: Annotated[int, Field(description="Exhaustiveness of the search (default is 8)")] = 8,
# ) -> str:
#     INPUT_DIR = f"{WDIR}/inputs"
#     os.makedirs(INPUT_DIR, exist_ok=True)
#     os.chdir(INPUT_DIR)

#     conda = os.environ.get("CONDA_EXE", "conda")
#     prefix = [conda, "run", "-n", "vina"]
#     cfg = "receptor_output.box.txt"

#     # prepare receptor
#     receptor_cmd = prefix + [
#         "mk_prepare_receptor.py",
#         "-i", receptor_path,
#         "-o", "receptor_output",
#         "-p",
#         "-v",
#         "--box_size", str(size_x), str(size_y), str(size_z),
#         "--box_center", str(center_x), str(center_y), str(center_z),
#     ]
#     print(" ".join(receptor_cmd))
#     p = subprocess.Popen(receptor_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     (output, err) = p.communicate()
#     print(output, err)

#     # prepare ligand
#     ligand_cmd = ["obabel", "-isdf", ligand_path, "-opdbqt", "-O", "ligand_output.pdbqt"]
#     print(" ".join(ligand_cmd))
#     p = subprocess.Popen(ligand_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     (output, err) = p.communicate()
#     print(output, err)

#     # dock molecule
#     vina_cmd = prefix + [
#         "vina",
#         "--receptor", "receptor_output.pdbqt",
#         "--ligand", "ligand_output.pdbqt",
#         "--config", cfg,
#         "--out", "docked.pdbqt",
#         "--exhaustiveness", str(exhaustiveness),
#     ]
#     print(" ".join(vina_cmd))
#     p = subprocess.Popen(vina_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     (output, err) = p.communicate()
#     print(output, err)
#     with open("docked.pdbqt", "r") as f:
#         content = f.read()
#         content = re.findall(r"REMARK VINA RESULT:\s+([-+]?\d*\.?\d+)", content)
#     binding_affinity = content[0]

#     return f"Successfully finished task. Protein pdbqt file location: {os.path.join(INPUT_DIR, "receptor_output.pdbqt")}\nDocked ligand pdbqt file location: {os.path.join(INPUT_DIR, "docked.pdbqt")}\nBinding Affinity: {binding_affinity}"


# @mcp.tool()
# def docked_protein_ligand_complex(
#     receptor_pdbqt_path: Annotated[str, Field(description="Receptor pdbqt file")],
#     ligand_pdbqt_path: Annotated[str, Field(description="Docked ligand pdbqt file")],
# ) -> str:
#     INPUT_DIR = f"{WDIR}/inputs"
#     os.makedirs(INPUT_DIR, exist_ok=True)
#     os.system(f"cp {receptor_pdbqt_path} {INPUT_DIR}/receptor.pdbqt")
#     os.system(f"cp {ligand_pdbqt_path} {INPUT_DIR}/ligand.pdbqt")
#     os.chdir(INPUT_DIR)

#     command = f"{PYTHON["diffusion"]} {combine_protein_ligand_file} -r {INPUT_DIR}/receptor.pdbqt -l {INPUT_DIR}/ligand.pdbqt -o {INPUT_DIR}/protein_ligand_complex.pdbqt"
#     p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     (output, err) = p.communicate()

#     command = f"obabel {INPUT_DIR}/protein_ligand_complex.pdbqt -O {INPUT_DIR}/protein_ligand_complex.pdb"
#     p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     (output, err) = p.communicate()
    
#     return f"Docked protein-ligand pdb file generated at: {INPUT_DIR}/protein_ligand_complex.pdb"


@mcp.tool()
def run_rf_diffusion(
    input_pdb: Annotated[str, Field(description="Input PDB file (both protein and ligand should be in this file)")],
    ligand: Annotated[str, Field(description="Ligand name")],
    interacting_atoms: Annotated[list[str], Field(description="Interacting atoms (in the protein), example if interacting atom is only A15, then [\"A15-15\"], otherwise if they are A15, A16, A17, then [\"A15-17\"], or if they are A15, A16, A17, A20, A21, then [\"A15-17, A20-21\"]")],
    N_designs: Annotated[int, Field(description="Number of designs to generate")] = 5,
    T_steps: Annotated[int, Field(description="Number of diffusion steps")] = 200,
    job_time: Annotated[str, Field(description="Time limit for diffusion jobs (estimated time is 3.5 * T_steps * N_design seconds)")] = "1:00:00"
) -> str:
    DIFFUSION_DIR = f"{WDIR}/0_diffusion"
    os.makedirs(DIFFUSION_DIR, exist_ok=True)
    os.system(f"rm -rf {DIFFUSION_DIR}/*")
    os.chdir(DIFFUSION_DIR)

    for interacting_atom in interacting_atoms:
        if "-" not in interacting_atom:
            return "Missing '-' in interacting atoms: follow the form A15-15 for A15 or A15-17 for A15 to A17 or A15-17, A20-21 for A15 to A17 and A20 to A21"

    residues = f"[\"0-120,{",0-120,".join(interacting_atoms)},0-120\"]"
    config = RF_DIFFUSION_CONFIG.format(
        T_steps=T_steps,
        N_designs=N_designs,
        LIGAND=ligand,
        residues=residues
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
        N_cores=2, 
        time=job_time, 
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
    diffusion_output_files = []
    with open(os.path.join(DIFFUSION_DIR, "output.log"), "r") as f:
        logs = f.read()
    with open(os.path.join(DIFFUSION_DIR, "output.err"), "r") as f:
        errors = f.read()
    for rundir in diffusion_rundirs:
        diffusion_outputs = glob.glob(os.path.join(DIFFUSION_DIR, f"{rundir}/out/*.pdb"))
        diffusion_output_files += diffusion_outputs
        for out in diffusion_outputs:
            with open(out, "r") as f:
                output_preds.append(f"File: {os.path.join(DIFFUSION_DIR, out)}\n\n" + f.read())
        with open(os.path.join(rundir, "output.log"), "r") as f:
            rfa_logs.append(f"File: {os.path.join(DIFFUSION_DIR, rundir, 'output.log')}\n\n" + f.read())
            
    # return f"RFDiffusionAA Job Completed\nAll output files: {diffusion_output_files}\n\n----------\n" + "\n----------\n----------\n".join(output_preds) + "\n----------\n\n" + "Log File:\n\n----------\n" + logs + "\n----------\n\n" + "RFDiffusionAA Log Files:\n\n----------\n" + "\n----------\n----------\n".join(rfa_logs) + "\n----------\n\n" + "Error File:\n\n----------\n" + errors + "\n----------\n"
    # return f"RFDiffusionAA Job Completed\nAll output files: {diffusion_output_files}\n\n" + "Log File:\n\n----------\n" + logs + "\n----------\n\n" + "RFDiffusionAA Log Files:\n\n----------\n" + "\n----------\n----------\n".join(rfa_logs) + "\n----------\n\n" + "Error File:\n\n----------\n" + errors + "\n----------\n"
    return f"RFDiffusionAA Job Completed\nAll output files: {diffusion_output_files}\n\n" + "Log File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


@mcp.tool()
def analyze_rf_diffusion_outputs(
    diffusion_outputs: Annotated[list[str], Field(description="List of diffusion output files")],
    ref_pdb: Annotated[str, Field(description="Reference PDB file (same as input PDB file)")],
    params: Annotated[list[str], Field(description="Rosetta params file(s) paths")],
    ref_catres: Annotated[str, Field(description="Position of interacting atom in diffusion input")] = None,
    term_limit: Annotated[float, Field(description="Cutoff for how close protein termini can be to the ligand")] = 15.0,
    SASA_limit: Annotated[float, Field(description="Cutoff for ligand relative SASA")] = 0.2,
    loop_limit: Annotated[float, Field(description="Cutoff for maximum allowed loop content")] = 0.3,
    longest_helix: Annotated[int, Field(description="Longest allowed helix length in AA's")] = 30,
    rog: Annotated[float, Field(description="Largest allowed radius of gyration")] = 30.0,
    exclude_clash_atoms: Annotated[str, Field(description="Ligand atom names that will be excluded from ligand clashchecking. Can be used for very flexible ligand parts")] = None,
    ligand_exposed_atoms: Annotated[str, Field(description="Ligand atoms that need to have SASA above cutoff defined with --exposed_atom_SASA. Used for selecting models where part of a ligand needs to be solvent-accessible")] = None,
    exposed_atom_SASA: Annotated[float, Field(description="Relative SASA cutoff for ligand atoms defined with ligand_exposed_atoms. Reasonable starting value would be 10.0, and then optimize from there")] = None,
    job_time: Annotated[str, Field(description="Time limit for diffusion jobs")] = "1:00:00"
) -> str:
    DIFFUSION_DIR = f"{WDIR}/0_diffusion"
    analysis_script = f"{SCRIPT_DIR}/scripts/diffusion_analysis/process_diffusion_outputs.py"
    os.chdir(DIFFUSION_DIR)
    os.system(f"rm -rf {DIFFUSION_DIR}/filtered_structures/*")

    dif_analysis_cmd_dict = {
        "--pdb": " ".join(diffusion_outputs),
        "--ref": ref_pdb,
        "--params": " ".join(params),
        "--rethread": True,
        "--fix": True,
        "--partial": None,
        "--outdir": None,
        "--traj": "5/30",
        "--trb": None,
        "--analyze": False,
    }
    
    # Add optional parameters only if they are not None
    if term_limit is not None:
        dif_analysis_cmd_dict["--term_limit"] = str(term_limit)
    if SASA_limit is not None:
        dif_analysis_cmd_dict["--SASA_limit"] = str(SASA_limit)
    if loop_limit is not None:
        dif_analysis_cmd_dict["--loop_limit"] = str(loop_limit)
    if ref_catres is not None:
        dif_analysis_cmd_dict["--ref_catres"] = ref_catres
    if exclude_clash_atoms is not None:
        dif_analysis_cmd_dict["--exclude_clash_atoms"] = exclude_clash_atoms
    if ligand_exposed_atoms is not None and exposed_atom_SASA is not None:
        dif_analysis_cmd_dict["--ligand_exposed_atoms"] = ligand_exposed_atoms
        dif_analysis_cmd_dict["--exposed_atom_SASA"] = str(exposed_atom_SASA)
    if longest_helix is not None:
        dif_analysis_cmd_dict["--longest_helix"] = str(longest_helix)
    if rog is not None:
        dif_analysis_cmd_dict["--rog"] = str(rog)

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
    # print(f"Analysis command: {analysis_command}")

    submit_script = "submit_diffusion_analysis.sh"
    utils.create_slurm_submit_script(
        filename=submit_script, 
        N_cores=2, 
        time=job_time, 
        command=analysis_command, 
        outfile_name="output_analysis"
    )
    p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("Heme Binder RFDiffusion Analysis Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(60)

    with open(os.path.join(DIFFUSION_DIR, "output_analysis.log"), "r") as f:
        logs = f.read()
    with open(os.path.join(DIFFUSION_DIR, "output_analysis.err"), "r") as f:
        errors = f.read()

    diffused_backbones_good = glob.glob(f"{DIFFUSION_DIR}/filtered_structures/*.pdb")
    dif_analysis_df = pd.read_csv(f"{DIFFUSION_DIR}/diffusion_analysis.sc", header=0, sep=r"\s+")
    # print(dif_analysis_df)

    if len(diffused_backbones_good) > 0:
        plt.figure(figsize=(12, 12))
        for i,k in enumerate(dif_analysis_df.keys()):
            if k in ["description"]:
                continue
            plt.subplot(4, 3, i+1)
            plt.hist(dif_analysis_df[k])
            plt.title(k)
            plt.xlabel(k)
        plt.tight_layout()
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, "outputs_0_analyze.png")
        plt.savefig(img_path)
        plt.close()
    else:
        img_path = "No image generated"

    return f"RFDiffusion Structures Filtered\nNumber of good backbone structure(s): {len(diffused_backbones_good)}\nAll output files: {diffused_backbones_good}\n\nDistribution plot path: {img_path}\n\nAnalysis CSV:\n\n----------\n" + dif_analysis_df.to_csv(index=False) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


@mcp.tool()
def run_protein_mpnn(
    diffused_backbones_good: Annotated[list[str], Field(description="List of good diffused backbone PDBs")],
    MPNN_omit_AAs: Annotated[str, Field(description="Amino acids to omit")] = "CM",
    MPNN_temperatures: Annotated[list, Field(description="List of temperatures for ProteinMPNN")] = [0.1, 0.2, 0.3],
    MPNN_outputs_per_temperature: Annotated[int, Field(description="Number of outputs per temperature")] = 5,
    job_time: Annotated[str, Field(description="Time limit for ProteinMPNN jobs")] = "1:00:00"
):
    MPNN_DIR = f"{WDIR}/1_proteinmpnn"
    if len(diffused_backbones_good) == 0:
        return "Error: No good backbones!"
    os.makedirs(MPNN_DIR, exist_ok=True)
    os.chdir(MPNN_DIR)
    os.system(f"rm -rf {MPNN_DIR}/*")

    mask_json_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/design/make_maskdict_from_trb.py --out masked_pos.jsonl --trb"
    for d in diffused_backbones_good:
        mask_json_cmd += " " + d.replace(".pdb", ".trb")
    p = subprocess.Popen(mask_json_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    json_creation_logs = output.decode('utf-8')
    json_creation_errors = err.decode('utf-8')
    if not os.path.exists("masked_pos.jsonl"):
        return "Error: Failed to create masked positions JSONL file\n\nLog File:\n\n----------\n" + json_creation_logs + "\n----------\n\nError File:\n\n----------\n" + json_creation_errors + "\n----------\n"

    commands_mpnn = []
    cmds_filename_mpnn = "commands_mpnn"
    with open(cmds_filename_mpnn, "w") as file:
        for T in MPNN_temperatures:
            for f in diffused_backbones_good:
                commands_mpnn.append(f"{PYTHON['proteinMPNN']} {proteinMPNN_script} "
                                    f"--model_type protein_mpnn --ligand_mpnn_use_atom_context 0 "
                                    "--fixed_residues_multi masked_pos.jsonl --out_folder ./ "
                                    f"--number_of_batches {MPNN_outputs_per_temperature} --temperature {T} "
                                    f"--omit_AA {MPNN_omit_AAs} --pdb_path {f} "
                                    f"--checkpoint_protein_mpnn {SCRIPT_DIR}/lib/LigandMPNN/model_params/proteinmpnn_v_48_020.pt\n")
                file.write(commands_mpnn[-1])
    # print("Example MPNN command:", commands_mpnn[-1])

    submit_script = "submit_mpnn.sh"
    utils.create_slurm_submit_script(
        filename=submit_script,
        N_cores=2,
        time=job_time,
        array=len(commands_mpnn),
        array_commandfile=cmds_filename_mpnn,
        group=10
    )
    p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("Heme Binder ProteinMPNN Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(60)

    with open(os.path.join(MPNN_DIR, "output.log"), "r") as f:
        logs = f.read()
    with open(os.path.join(MPNN_DIR, "output.err"), "r") as f:
        errors = f.read()
    sequences = []
    for s in glob.glob(f"{MPNN_DIR}/seqs/*.fa"):
        with open(s, "r") as f:
            seq = f.read()
        sequences.append(f"File: {s}\n\n" + seq)
    backbones = []
    for b in glob.glob(f"{MPNN_DIR}/backbones/*.pdb"):
        with open(b, "r") as f:
            pdb = f.read()
        backbones.append(f"File: {b}\n\n" + pdb)
    protein_seq_fasta_files = glob.glob(f"{MPNN_DIR}/seqs/*.fa")
    
    # return f"ProteinMPNN Job Completed\nAll output files: {protein_seq_fasta_files}\n\nSequences:\n\n----------\n" + "\n----------\n----------\n".join(sequences) + "\n----------\n\nBackbones:\n\n----------\n" + "\n----------\n----------\n".join(backbones) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n\nMasked positions JSONL file creation logs:\n\n----------\n" + json_creation_logs + "\n----------\n\nMasked positions JSONL file creation errors:\n\n----------\n" + json_creation_errors + "\n----------\n"
    # return f"ProteinMPNN Job Completed\nAll output files: {protein_seq_fasta_files}\n\n----------\n" + "\n----------\n----------\n".join(sequences) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n\nMasked positions JSONL file creation logs:\n\n----------\n" + json_creation_logs + "\n----------\n\nMasked positions JSONL file creation errors:\n\n----------\n" + json_creation_errors + "\n----------\n"
    return f"ProteinMPNN Job Completed\nAll output files: {protein_seq_fasta_files}\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n\nMasked positions JSONL file creation logs:\n\n----------\n" + json_creation_logs + "\n----------\n\nMasked positions JSONL file creation errors:\n\n----------\n" + json_creation_errors + "\n----------\n"


@mcp.tool()
def run_af2(
    protein_seq_fasta_files: Annotated[list[str], Field(description="Protein sequence fasta files (ProteinMPNN outputs)")],
    after_ligand_mpnn: Annotated[bool, Field(description="Running AF2 after ligand MPNN (True) or Protein MPNN (False)")] = False,
    AF2_recycles: Annotated[int, Field(description="Number of AF2 recycling steps")] = 3,
    AF2_models: Annotated[str, Field(description="Number of AF2 models (default is '4', add other models to this string if needed, i.e. '3 4 5')")] = "4",
    job_time: Annotated[str, Field(description="Time limit for AF2 jobs")] = "1:00:00",
):
    if after_ligand_mpnn:
        AF2_DIR = f"{WDIR}/5.1_2nd_af2"
    else:
        AF2_DIR = f"{WDIR}/2_af2"
    os.makedirs(AF2_DIR, exist_ok=True)
    os.chdir(AF2_DIR)
    os.system(f"rm -rf {AF2_DIR}/*")

    ### First collecting MPNN outputs and creating FASTA files for AF2 input
    mpnn_fasta = utils.parse_fasta_files(protein_seq_fasta_files)

    if after_ligand_mpnn:
        # Giving sequences unique names based on input PDB name, temperature, and sequence identifier
        _mpnn_fasta = {}
        for k, seq in mpnn_fasta.items():
            if "model_path" in k:
                _mpnn_fasta[k.split(",")[0]+"_native"] = seq.strip()
            else:
                _mpnn_fasta[k.split(",")[0]+"_"+k.split(",")[2].replace(" T=", "T")+"_0_"+k.split(",")[1].replace(" id=", "")] = seq.strip()
        mpnn_fasta = {k:v for k,v in _mpnn_fasta.items()}
    else:
        mpnn_fasta = {k: seq.strip() for k, seq in mpnn_fasta.items() if "model_path" not in k}
        # Giving sequences unique names based on input PDB name, temperature, and sequence identifier
        mpnn_fasta = {k.split(",")[0]+"_"+k.split(",")[2].replace(" T=", "T")+"_0_"+k.split(",")[1].replace(" id=", ""): seq for k, seq in mpnn_fasta.items()}
    # print(f"A total on {len(mpnn_fasta)} sequences will be predicted.")

    SEQUENCES_PER_AF2_JOB = 100
    mpnn_fasta_split = utils.split_fasta_based_on_length(mpnn_fasta, SEQUENCES_PER_AF2_JOB, write_files=True)

    commands_af2 = []
    cmds_filename_af2 = "commands_af2"
    with open(cmds_filename_af2, "w") as file:
        for ff in glob.glob("*.fasta"):
            commands_af2.append(f"{PYTHON['af2']} {AF2_script} "
                                f"--af-nrecycles {AF2_recycles} --af-models {AF2_models} "
                                f"--fasta {ff} --scorefile {ff.replace('.fasta', '.csv')}\n")
            file.write(commands_af2[-1])
    # print("Example AF2 command:", commands_af2[-1])

    submit_script = "submit_af2.sh"
    utils.create_slurm_submit_script(
        filename=submit_script, 
        N_cores=2, 
        time=job_time, 
        array=len(commands_af2),
        array_commandfile=cmds_filename_af2
    )
    
    with open(submit_script, "r") as f:
        content = f.read()
        content = content.split("\n")
        content.insert(-2, "module load cuda/12.6")
    with open(submit_script, "w") as f:
        f.write("\n".join(content))

    p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("Heme Binder AF2 Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(60)

    with open(os.path.join(AF2_DIR, "output.log"), "r") as f:
        logs = f.read()
    with open(os.path.join(AF2_DIR, "output.err"), "r") as f:
        errors = f.read()
    structures = []
    for b in glob.glob(f"{AF2_DIR}/*.pdb"):
        with open(b, "r") as f:
            pdb = f.read()
        structures.append(f"File: {b}\n\n" + pdb)
    with open(glob.glob(f"{AF2_DIR}/*.fasta")[0], "r") as f:
        input_sequences = f.read()
    af2_csv_files = glob.glob(f"{AF2_DIR}/*.csv")
    with open(af2_csv_files[0], "r") as f:
        af2_scores_csv = f.read()
    af2_out_files = glob.glob(f"{AF2_DIR}/*.pdb")
    
    # return f"AF2 Job Completed\nAll output files: {af2_out_files}\n\n----------\n" + "\n----------\n----------\n".join(structures) + "\n----------\n\nInput Sequences:\n\n----------\n" + input_sequences + f"\n----------\n\nAF2 Scores CSVs: {af2_csv_files}\nExample:\n\n----------\n" + af2_scores_csv + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"
    # return f"AF2 Job Completed\nAll output files: {af2_out_files}\n\n----------\n" + "\n----------\n----------\n".join(structures) + f"\n----------\n\nAF2 Scores CSVs: {af2_csv_files}\nExample:\n\n----------\n" + af2_scores_csv + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"
    return f"AF2 Job Completed\nAll output files: {af2_out_files}\n\nAF2 Scores CSVs: {af2_csv_files}\n\n----------\n" + af2_scores_csv + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


@mcp.tool()
def analyze_af2_outputs(
    params: Annotated[list[str], Field(description="Rosetta params file(s) paths")],
    ref_pdbs_path: Annotated[str, Field(description="Path to reference PDB file(s) (RFDiffusionAA filtered outputs folder or LigandMPNN good designs folder)")],
    after_ligand_mpnn: Annotated[bool, Field(description="Running AF2 after ligand MPNN (True) or Protein MPNN (False)")] = False,
    af2_output_csv_paths: Annotated[list[str], Field(description="AF2 output CSV file(s) paths (default is all CSV files in the AF2 directory)")] = None,
    job_time: Annotated[str, Field(description="Time limit for AF2 analysis jobs")] = "1:00:00",
    lDDT: Annotated[float, Field(description="lDDT minimum cutoff")] = 70.0,
    rmsd: Annotated[float, Field(description="rmsd maximum cutoff")] = 5.0,
    rmsd_SR1: Annotated[float, Field(description="rmsd_SR1 maximum cutoff")] = 10.0
):
    if after_ligand_mpnn:
        AF2_DIR = f"{WDIR}/5.1_2nd_af2"
    else:
        AF2_DIR = f"{WDIR}/2_af2"
    os.chdir(AF2_DIR)
    os.system("rm -rf scores.sc")
    os.system("rm -rf filtered_scores.sc")
    os.system("rm -rf good/*")

    if af2_output_csv_paths is None:
        os.system("head -n 1 $(ls *aa*.csv | shuf -n 1) > scores.csv ; for f in *aa*.csv ; do tail -n +2 ${f} >> scores.csv ; done")
        if not os.path.exists("scores.csv"):
            return "Error: CSV file not found"
    else:
        df = pd.DataFrame()
        for csv in af2_output_csv_paths:
            df = pd.concat([df, pd.read_csv(csv)], ignore_index=True)
        df.to_csv("scores.csv", index=False)

    ### Calculating the RMSDs of AF2 predictions relative to the diffusion outputs
    ### Catalytic residue sidechain RMSDs are calculated in the reference PDB has REMARK 666 line present
    analysis_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/utils/analyze_af2.py --scorefile scores.csv --ref_path {ref_pdbs_path} --mpnn --params {' '.join(params)}"
    submit_script = "submit_af2_analysis.sh"
    utils.create_slurm_submit_script(
        filename=submit_script, 
        N_cores=2, 
        time=job_time, 
        command=analysis_cmd, 
        outfile_name="output_analysis"
    )

    p = subprocess.Popen(["sbatch", submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("Heme Binder AF2 Analysis Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(60)

    logs = open(os.path.join(AF2_DIR, "output_analysis.log"), "r").read()
    errors = open(os.path.join(AF2_DIR, "output_analysis.err"), "r").read()

    scores_af2 = pd.read_csv("scores.sc", sep=r"\s+", header=0)

    AF2_filters = {
        "lDDT": [lDDT, ">="],
        "rmsd": [rmsd, "<="],
        "rmsd_SR1": [rmsd_SR1, "<="]
    }
    scores_af2_filtered = utils.filter_scores(scores_af2, AF2_filters)
    utils.dump_scorefile(scores_af2_filtered, "filtered_scores.sc")

    plt.figure(figsize=(12, 3))
    for i,k in enumerate(AF2_filters):
        plt.subplot(1, 3, i+1)
        plt.hist(scores_af2[k])
        plt.title(k)
        plt.xlabel(k)
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    if after_ligand_mpnn:
        img_path = os.path.join(output_dir, "outputs_5_analyze.png")
    else:
        img_path = os.path.join(output_dir, "outputs_2_analyze.png")
    plt.savefig(img_path)
    plt.close()

    if len(scores_af2_filtered) > 0:
        os.makedirs("good", exist_ok=True)
        os.system(f"rm -rf {AF2_DIR}/good/*")
        good_af2_models = [row["Output_PDB"]+".pdb" for idx,row in scores_af2_filtered.iterrows()]
        for pdb in good_af2_models:
            copy2(pdb, f"good/{pdb}")
        good_af2_models = glob.glob(f"{AF2_DIR}/good/*.pdb")
    else:
        return f"Error: No good models to continue this pipeline with\n\nCSV File:\n\n----------\n{scores_af2.to_csv(index=False)}\n----------\n\nLogs:\n\n----------\n" + logs + "\n----------\n\nErrors:\n\n----------\n" + errors + "\n----------\n"

    if after_ligand_mpnn:
        af2_good_files = glob.glob(f"{AF2_DIR}/good/*.pdb")
        return f"AF2 Designs Filtered\nNumber of good structure(s): {len(good_af2_models)}\nAF2 analysis filtered output files: {af2_good_files}\n\nAF2 Scores Distribution Plot: {img_path}\n\nFiltered CSV File:\n\n----------\n" + scores_af2_filtered.to_csv(index=False) + "\n----------\n\nAF2 Analysis Log File:\n\n----------\n" + logs + "\n----------\n\nAF2 Analysis Error File:\n\n----------\n" + errors + "\n----------\n"

    os.chdir(f"{AF2_DIR}/good")
    ref_pdb_files = glob.glob(f"{ref_pdbs_path}/*.pdb")
    align_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/utils/place_ligand_after_af2.py "\
                f"--outdir with_heme2 --params {' '.join(params)} --fix_catres "\
                f"--pdb {' '.join(good_af2_models)} "\
                f"--ref {' '.join(ref_pdb_files)}"
    p = subprocess.Popen(align_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (alignment_logs, alignment_errors) = p.communicate()

    final_output = []
    for pdb in glob.glob(f"{AF2_DIR}/good/with_heme2/*.pdb"):
        with open(pdb, "r") as f:
            pdb_content = f.read()
        final_output.append(f"File: {pdb}\n\n" + pdb_content)

    # return f"AF2 Designs Filtered\nNumber of good structure(s): {len(good_af2_models)}\nAF2 analysis filtered output files: {good_af2_models}\nAF2 analysis filtered and aligned output files: {glob.glob(f"{AF2_DIR}/good/with_heme2/*.pdb")}\n\nAF2 Scores Distribution Plot: {img_path}\n\n----------\n" + "\n----------\n----------\n".join(final_output) + "\n----------\n\nCSV File:\n\n----------\n" + scores_af2.to_csv(index=False) + "\n----------\n\nFiltered CSV File:\n\n----------\n" + scores_af2_filtered.to_csv(index=False) + "\n----------\n\nAF2 Analysis Log File:\n\n----------\n" + logs + "\n----------\n\nAF2 Analysis Error File:\n\n----------\n" + errors + "\n----------\n\nAlignment Log File:\n\n----------\n" + alignment_logs.decode('utf-8') + "\n----------\n\nAlignment Error File:\n\n----------\n" + alignment_errors.decode('utf-8') + "\n----------\n"
    # return f"AF2 Designs Filtered\nNumber of good structure(s): {len(good_af2_models)}\nAF2 analysis filtered output files: {good_af2_models}\nAF2 analysis filtered and aligned output files: {glob.glob(f"{AF2_DIR}/good/with_heme2/*.pdb")}\n\nAF2 Scores Distribution Plot: {img_path}\n\nCSV File:\n\n----------\n" + scores_af2.to_csv(index=False) + "\n----------\n\nFiltered CSV File:\n\n----------\n" + scores_af2_filtered.to_csv(index=False) + "\n----------\n\nAF2 Analysis Log File:\n\n----------\n" + logs + "\n----------\n\nAF2 Analysis Error File:\n\n----------\n" + errors + "\n----------\n\nAlignment Log File:\n\n----------\n" + alignment_logs.decode('utf-8') + "\n----------\n\nAlignment Error File:\n\n----------\n" + alignment_errors.decode('utf-8') + "\n----------\n"
    af2_aligned_files = glob.glob(f"{AF2_DIR}/good/with_heme2/*.pdb")
    return f"AF2 Designs Filtered\nNumber of good structure(s): {len(good_af2_models)}\nAF2 analysis filtered and aligned output files: {af2_aligned_files}\n\nAF2 Scores Distribution Plot: {img_path}\n\nFiltered CSV File:\n\n----------\n" + scores_af2_filtered.to_csv(index=False) + "\n----------\n\nAF2 Analysis Log File:\n\n----------\n" + logs + "\n----------\n\nAF2 Analysis Error File:\n\n----------\n" + errors + "\n----------\n\nAlignment Log File:\n\n----------\n" + alignment_logs.decode('utf-8') + "\n----------\n\nAlignment Error File:\n\n----------\n" + alignment_errors.decode('utf-8') + "\n----------\n"


@mcp.tool()
def run_ligand_mpnn(
    input_pdbs: Annotated[list[str], Field(description="Input PDB files (aligned and filtered outputs of AF2 analysis)")],
    cstfile: Annotated[str, Field(description="Path to CST file")],
    params: Annotated[list[str], Field(description="Rosetta params file(s) paths")],
    # filters: Annotated[dict, Field(description="Filtering criteria for ligand MPNN designs - minimum requirements (example: {\"all_cst\": [1.5, \"<=\"], \"L_SASA\": [0.20, \"<=\"], \"COO_hbond\": [1.0, \"=\"], \"cms_per_atom\": [5.0, \">=\"], \"corrected_ddg\": [-50.0, \"<=\"], \"nlr_totrms\": [0.8, \"<=\"], \"nlr_SR1_rms\": [0.6, \"<=\"]})")] = {},
    align_atoms: Annotated[list[str], Field(description="Ligand atom names used for aligning the rotamers (example: ['N1', 'N2'])")] = ["N1", "N2", "N3", "N4"],
    NSTRUCT: Annotated[int, Field(description="Number of ligand MPNN designs")] = 5,
    job_time: Annotated[str, Field(description="Time limit for ligand MPNN jobs")] = "6:00:00"
):
    filters = {}
    DESIGN_DIR_ligMPNN = f"{WDIR}/3.1_design_pocket_ligandMPNN"
    os.makedirs(DESIGN_DIR_ligMPNN, exist_ok=True)
    os.system(f"rm -rf {DESIGN_DIR_ligMPNN}/*")
    os.makedirs(DESIGN_DIR_ligMPNN+"/logs", exist_ok=True)
    os.chdir(DESIGN_DIR_ligMPNN)

    for pdb in input_pdbs:
        if not pdb.endswith(".pdb"):
            return "Error: All input PDB files must end with .pdb"
    
    os.system(f"cp {SCRIPT_DIR}/scripts/design/scoring/heme_scoring.py {SCRIPT_DIR}/scripts/design/scoring/heme_scoring_new.py")
    with open(f"{SCRIPT_DIR}/scripts/design/scoring/heme_scoring_new.py", "a") as file:
        file.write(f"filters = {filters}\nalign_atoms = {align_atoms}\n")

    commands_design = []
    cmds_filename_des = "commands_design"
    with open(cmds_filename_des, "w") as file:
        for pdb in input_pdbs:
            commands_design.append(f"{PYTHON['general']} {SCRIPT_DIR}/scripts/design/heme_pocket_ligMPNN.py "
                                f"--pdb {pdb} --nstruct {NSTRUCT} "
                                f"--scoring {SCRIPT_DIR}/scripts/design/scoring/heme_scoring_new.py "
                                f"--params {' '.join(params)} --cstfile {cstfile} > logs/{os.path.basename(pdb).replace('.pdb', '.log')}\n")
            file.write(commands_design[-1])
    submit_script = "submit_design.sh"
    utils.create_slurm_submit_script(
        filename=submit_script, 
        N_cores=2, 
        time=job_time, 
        array=len(commands_design),
        array_commandfile=cmds_filename_des,
        partition="GPU-shared"
    )

    p = subprocess.Popen(["sbatch", submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("Heme Binder Ligand MPNN Design Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(300)

    logs = open(os.path.join(DESIGN_DIR_ligMPNN, "output.log"), "r").read()
    errors = open(os.path.join(DESIGN_DIR_ligMPNN, "output.err"), "r").read()

    pdb_files = glob.glob(f"{DESIGN_DIR_ligMPNN}/*.pdb")
    pdb_content = []
    for pdb in pdb_files:
        with open(pdb, "r") as f:
            pdb_content.append(f"File: {pdb}\n\n{f.read()}")
    lig_mpnn_logs = []
    for log_file in glob.glob(f"{DESIGN_DIR_ligMPNN}/logs/*.log"):
        with open(log_file, "r") as f:
            lig_mpnn_logs.append(f"File: {log_file}\n\n{f.read()}")
    if os.path.exists("scorefile.txt"):
        scores = pd.read_csv("scorefile.txt", sep=r"\s+", header=0)
    else:
        return "Error: Ligand MPNN ran into an error:\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nLigand MPNN Logs:\n\n----------\n" + "\n----------\n----------\n".join(lig_mpnn_logs) + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"
    
    # return f"LigandMPNN Job Completed\nAll output files: {pdb_files}\nScores File: {os.path.join(DESIGN_DIR_ligMPNN, "scorefile.txt")}\n\n----------\n" + "\n----------\n----------\n".join(pdb_content) + "\n----------\n\nScores file:\n\n----------\n" + scores.to_csv(index=False) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nLigand MPNN Logs:\n\n----------\n" + "\n----------\n----------\n".join(lig_mpnn_logs) + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"
    # return f"LigandMPNN Job Completed\nAll output files: {pdb_files}\nScores File: {os.path.join(DESIGN_DIR_ligMPNN, "scorefile.txt")}\n\n----------\n" + scores.to_csv(index=False) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nLigand MPNN Logs:\n\n----------\n" + "\n----------\n----------\n".join(lig_mpnn_logs) + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"
    return f"LigandMPNN Job Completed\nAll output files: {pdb_files}\nScores File: {os.path.join(DESIGN_DIR_ligMPNN, "scorefile.txt")}\n\n----------\n" + scores.to_csv(index=False) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


@mcp.tool()
def analyze_ligand_mpnn_outputs(
    scores_file: Annotated[str, Field(description="Scores file path (most often it will be the scorefile.txt from Ligand MPNN)")],
    filters: Annotated[dict, Field(description="Filtering criteria for ligand MPNN designs - based on the values in the scores file and minimum requirements (example: {\"all_cst\": [1.5, \"<=\"], \"L_SASA\": [0.20, \"<=\"], \"COO_hbond\": [1.0, \"=\"], \"cms_per_atom\": [5.0, \">=\"], \"corrected_ddg\": [-50.0, \"<=\"], \"nlr_totrms\": [0.8, \"<=\"], \"nlr_SR1_rms\": [0.6, \"<=\"]})")] = {},
):
    DESIGN_DIR_ligMPNN = f"{WDIR}/3.1_design_pocket_ligandMPNN"
    os.chdir(DESIGN_DIR_ligMPNN)
    os.system(f"rm -rf {DESIGN_DIR_ligMPNN}/good/*")

    scores = pd.read_csv(scores_file, sep=r"\s+", header=0)
    filtered_scores = utils.filter_scores(scores, filters)

    plt.figure(figsize=(12, 9))
    for i,k in enumerate(filters):
        plt.subplot(3, 3, i+1)
        plt.hist(scores[k])
        plt.title(k)
        plt.xlabel(k)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "outputs_3_analyze.png")
    plt.savefig(img_path)
    plt.close()

    if len(filtered_scores) > 0:
        os.makedirs(f"{DESIGN_DIR_ligMPNN}/good", exist_ok=True)
        for idx, row in filtered_scores.iterrows():
            copy2(row["description"]+".pdb", "good/"+row["description"]+".pdb")
    else:
        return "Error: No good designs created, change the filtering criteria or rerun the previous jobs with better parameters\n\nScores File:\n\n----------\n" + scores.to_csv(index=False) + "\n----------\n"

    # return f"Ligand MPNN Designs Filtered\nNumber of good structure(s): {len(filtered_scores)}\nAll output files: {glob.glob(f"{DESIGN_DIR_ligMPNN}/good/*.pdb")}\n\nFiltered Score File:\n\n----------\n{filtered_scores.to_csv(index=False)}\n----------\n\nOriginal Score File ({scores_file}):\n\n----------\n{scores.to_csv(index=False)}\n----------\n"
    # return f"Ligand MPNN Designs Filtered\nNumber of good structure(s): {len(filtered_scores)}\nAll output files: {glob.glob(f"{DESIGN_DIR_ligMPNN}/good/*.pdb")}\n\nFiltered Score File:\n\n----------\n{filtered_scores.to_csv(index=False)}\n----------\n\nOriginal Score File ({scores_file}):\n\n----------\n{scores.to_csv(index=False)}\n----------\n"
    ligmpnn_good_files = glob.glob(f"{DESIGN_DIR_ligMPNN}/good/*.pdb")
    return f"Ligand MPNN Designs Filtered\nNumber of good structure(s): {len(filtered_scores)}\nAll output files: {ligmpnn_good_files}\n\nFiltered Score File:\n\n----------\n{filtered_scores.to_csv(index=False)}\n----------\n"


@mcp.tool()
def run_ligand_mpnn_redesign_2nd_layer_residues(
    input_pdbs: Annotated[list[str], Field(description="Input PDB files paths")],
    LIGAND: Annotated[str, Field(description="Ligand name")],
    params: Annotated[list[str], Field(description="Rosetta params file(s) paths")],
    dist_bb: Annotated[float, Field(description="Distance threshold for backbone atoms")] = 6.0,
    dist_sc: Annotated[float, Field(description="Distance threshold for sidechain atoms")] = 5.0,
    MPNN_omit_AAs: Annotated[str, Field(description="Amino acids to omit")] = "CM",
    MPNN_temperatures: Annotated[list, Field(description="List of temperatures for LigandMPNN")] = [0.1, 0.2],
    MPNN_outputs_per_temperature: Annotated[int, Field(description="Number of outputs per temperature")] = 5,
    job_time: Annotated[str, Field(description="Time limit for the job")] = "0:15:00",
):
    DESIGN_DIR_2nd_mpnn = f"{WDIR}/4.1_2nd_mpnn"
    os.makedirs(DESIGN_DIR_2nd_mpnn, exist_ok=True)
    os.system(f"rm -rf {DESIGN_DIR_2nd_mpnn}/*")
    os.chdir(DESIGN_DIR_2nd_mpnn)
    
    if len(input_pdbs) == 0:
        return "Error: No designs to run 2nd MPNN on."
    for pdb in input_pdbs:
        if not pdb.endswith(".pdb"):
            return "Error: All input PDB files must end with .pdb"

    ### Making a JSON file specifiying designable positions for each structure.
    ### Will also make non-pocket ALA positions as designable.
    ### This is to fix any surface ALA-patches that previous MPNN may have introduced.
    make_json_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/design/setup_ligand_mpnn_2nd_layer.py "\
                    f"--params {' '.join(params)} --ligand {LIGAND} --output_path parsed_pdbs_lig.jsonl "\
                    f"--output_path masked_pos.jsonl --dist_bb {dist_bb} --dist_sc {dist_sc} "\
                    f"--pdb {' '.join(input_pdbs)}"
    p = subprocess.Popen(make_json_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    json_file_creation_logs = output.decode('utf-8')
    json_file_creation_errors = err.decode('utf-8')
    if not os.path.exists(os.path.join("masked_pos.jsonl")):
        return "Error: Failed to create masked positions JSONL file\n\nJSON File (designable positions) Creation Logs:\n\n----------\n" + json_file_creation_logs + "\n----------\n\nJSON File (designable positions) Creation Errors:\n\n----------\n" + json_file_creation_errors + "\n----------\n"

    commands_mpnn = []
    cmds_filename_mpnn = "commands_mpnn"
    with open(cmds_filename_mpnn, "w") as file:
        for T in MPNN_temperatures:
            for f in input_pdbs:
                commands_mpnn.append(f"{PYTHON['proteinMPNN']} {proteinMPNN_script} "
                                    f"--model_type ligand_mpnn --ligand_mpnn_use_atom_context 1 "
                                    "--fixed_residues_multi masked_pos.jsonl --out_folder ./ "
                                    f"--number_of_batches {MPNN_outputs_per_temperature} --temperature {T} "
                                    f"--omit_AA {MPNN_omit_AAs} --pdb_path {f} "
                                    f"--checkpoint_ligand_mpnn {SCRIPT_DIR}/lib/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt\n")
                file.write(commands_mpnn[-1])
    
    submit_script = "submit_mpnn.sh"
    utils.create_slurm_submit_script(
        filename=submit_script, 
        N_cores=2, 
        time=job_time, 
        array=len(commands_mpnn),
        array_commandfile=cmds_filename_mpnn, 
        group=10
    )
    p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("2nd Ligand MPNN Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(60)

    with open(os.path.join(DESIGN_DIR_2nd_mpnn, "output.log"), "r") as f:
        logs = f.read()
    with open(os.path.join(DESIGN_DIR_2nd_mpnn, "output.err"), "r") as f:
        errors = f.read()
    sequences = []
    for s in glob.glob(f"{DESIGN_DIR_2nd_mpnn}/seqs/*.fa"):
        with open(s, "r") as f:
            seq = f.read()
        sequences.append(f"File: {s}\n\n" + seq)
    backbones = []
    for b in glob.glob(f"{DESIGN_DIR_2nd_mpnn}/backbones/*.pdb"):
        with open(b, "r") as f:
            pdb = f.read()
        backbones.append(f"File: {b}\n\n" + pdb)
    protein_seq_fasta_files = glob.glob(f"{DESIGN_DIR_2nd_mpnn}/seqs/*.fa")
    
    # return f"LigandMPNN Job Completed\nAll output files: {protein_seq_fasta_files}\n\nSequences:\n\n----------\n" + "\n----------\n----------\n".join(sequences) + "\n----------\n\nBackbones:\n\n----------\n" + "\n----------\n----------\n".join(backbones) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n\nJSON File (designable positions) Creation Logs:\n\n----------\n" + json_file_creation_logs + "\n----------\n\nJSON File (designable positions) Creation Errors:\n\n----------\n" + json_file_creation_errors + "\n----------\n"
    # return f"LigandMPNN Job Completed\nAll output files: {protein_seq_fasta_files}\n\n----------\n" + "\n----------\n----------\n".join(sequences) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n\nJSON File (designable positions) Creation Logs:\n\n----------\n" + json_file_creation_logs + "\n----------\n\nJSON File (designable positions) Creation Errors:\n\n----------\n" + json_file_creation_errors + "\n----------\n"
    return f"LigandMPNN Job Completed\nAll output files: {protein_seq_fasta_files}\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n\nJSON File (designable positions) Creation Logs:\n\n----------\n" + json_file_creation_logs + "\n----------\n\nJSON File (designable positions) Creation Errors:\n\n----------\n" + json_file_creation_errors + "\n----------\n"


@mcp.tool()
def run_fast_relax(
    input_pdbs: Annotated[list[str], Field(description="Paths to input PDB file(s) (from AF2 filtered good models)")],
    ref_pdbs: Annotated[list[str], Field(description="Paths to reference PDB file(s) (make sure to give all the structures from Ligand MPNN good pocket designs)")],
    cstfile: Annotated[str, Field(description="Path to CST file")],
    LIGAND: Annotated[str, Field(description="Ligand name")],
    params: Annotated[list[str], Field(description="Rosetta params file(s) paths")],
    NSTRUCT: Annotated[int, Field(description="Number of relax iteration on each input structure")] = 1,
    job_time: Annotated[str, Field(description="Time limit for the job")] = "0:30:00"
):
    RELAX_DIR = f"{WDIR}/6.1_final_relax"
    os.makedirs(RELAX_DIR, exist_ok=True)
    os.system(f"rm -rf {RELAX_DIR}/*")
    os.makedirs(RELAX_DIR+"/logs", exist_ok=True)
    os.chdir(RELAX_DIR)

    if len(input_pdbs) == 0:
        return "Error: No input PDB files to relax with"
    for pdb in input_pdbs:
        if not pdb.endswith(".pdb"):
            return "Error: All input PDB files must end with .pdb"
    
    ref_and_model_pairs = []
    for r in ref_pdbs:
        for pdbfile in input_pdbs:
            if os.path.basename(r).replace(".pdb", "_") in pdbfile:
                ref_and_model_pairs.append((r, pdbfile))

    if len(ref_and_model_pairs) != len(input_pdbs): 
        return "Error: Was not able to match all models with reference structures"

    commands_relax = []
    cmds_filename_rlx = "commands_relax"
    with open(cmds_filename_rlx, "w") as file:
        for r_m in ref_and_model_pairs:
            commands_relax.append(f"{PYTHON['general']} {SCRIPT_DIR}/scripts/design/align_add_ligand_relax.py "
                                f"--outdir ./ --ligand {LIGAND} --ref_pdb {r_m[0]} "
                                f"--pdb {r_m[1]} --nstruct {NSTRUCT} "
                                f"--params {' '.join(params)} --cstfile {cstfile} > logs/{os.path.basename(r_m[1]).replace('.pdb', '.log')}\n")
            file.write(commands_relax[-1])

    submit_script = "submit_relax.sh"
    utils.create_slurm_submit_script(
        filename=submit_script, 
        N_cores=2, 
        time=job_time, 
        array=len(commands_relax),
        array_commandfile=cmds_filename_rlx
    )

    p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    print(output.decode('utf-8'), err.decode('utf-8'))
    job_id = extract_job_id(output.decode('utf-8'))
    print("Fast Relax Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(60)

    with open(os.path.join(RELAX_DIR, "output.log"), "r") as f:
        logs = f.read()
    with open(os.path.join(RELAX_DIR, "output.err"), "r") as f:
        errors = f.read()
    pdb_files = glob.glob(f"{RELAX_DIR}/*.pdb")
    pdb_content = []
    for pdb in pdb_files:
        with open(pdb, "r") as f:
            pdb_content.append(f"File: {pdb}\n\n{f.read()}")
    relax_logs = []
    for log_file in glob.glob(f"{RELAX_DIR}/logs/*.log"):
        with open(log_file, "r") as f:
            relax_logs.append(f"File: {log_file}\n\n{f.read()}")
    if os.path.exists("scorefile.txt"):
        scores = pd.read_csv("scorefile.txt", sep=r"\s+", header=0)
    else:
        return "Fast Relax job ran into an error\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"

    # return f"Fast Relax Job Completed\nAll output files: {pdb_files}\nScores File: {os.path.join(RELAX_DIR, "scorefile.txt")}\n\nPDB Files:\n\n----------\n" + "\n----------\n----------\n".join(pdb_content) + "\n----------\n\nScores File:\n\n----------\n" + scores.to_csv(index=False) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nFast Relax Logs:\n\n----------\n" + "\n----------\n----------\n".join(relax_logs) + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"
    # return f"Fast Relax Job Completed\nAll output files: {pdb_files}\nScores File: {os.path.join(RELAX_DIR, "scorefile.txt")}\n\n----------\n" + scores.to_csv(index=False) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nFast Relax Logs:\n\n----------\n" + "\n----------\n----------\n".join(relax_logs) + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"
    return f"Fast Relax Job Completed\nAll output files: {pdb_files}\nScores File: {os.path.join(RELAX_DIR, "scorefile.txt")}\n\n----------\n" + scores.to_csv(index=False) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


@mcp.tool()
def analyze_fast_relax_outputs(
    scores_file: Annotated[str, Field(description="Scores file path (most often it will be the scorefile.txt from Fast Relax)")],
    filters: Annotated[dict, Field(description="Filtering criteria for fast relax designs - based on the values in the scores file and minimum requirements (example: {\"all_cst\": [1.5, \"<=\"], \"L_SASA\": [0.20, \"<=\"], \"COO_hbond\": [1.0, \"=\"], \"cms_per_atom\": [5.0, \">=\"], \"corrected_ddg\": [-50.0, \"<=\"], \"nlr_totrms\": [0.8, \"<=\"], \"nlr_SR1_rms\": [0.6, \"<=\"]})")] = {},
) -> str:
    RELAX_DIR = f"{WDIR}/6.1_final_relax"
    os.chdir(RELAX_DIR)
    os.system("rm -rf good/*")

    scores = pd.read_csv(scores_file, sep=r"\s+", header=0)
    filtered_scores = utils.filter_scores(scores, filters)

    plt.figure(figsize=(12, 9))
    for i,k in enumerate(filters):
        if k not in scores.keys():
            continue
        plt.subplot(4, 3, i+1)
        plt.hist(scores[k])
        plt.title(k)
        plt.xlabel(k)
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "outputs_6_analyze.png")
    plt.savefig(img_path)
    plt.close()

    if len(filtered_scores) > 0:
        os.makedirs(f"{RELAX_DIR}/good", exist_ok=True)
        for idx, row in filtered_scores.iterrows():
            copy2(row["description"]+".pdb", "good/"+row["description"]+".pdb")
    else:
        return "Error: No good designs created, change the filtering criteria or rerun the previous jobs with better parameters\n\nScores File:\n\n----------\n" + scores.to_csv(index=False) + "\n----------\n"
    
    # return f"Fast Relax Designs Filtered\nNumber of good structure(s): {len(filtered_scores)}\nAll output files: {glob.glob(f"{RELAX_DIR}/good/*.pdb")}\n\nFiltered Score File:\n\n----------\n{filtered_scores.to_csv(index=False)}\n----------\n\nOriginal Score File ({scores_file}):\n\n----------\n{scores.to_csv(index=False)}\n----------\n"
    # return f"Fast Relax Designs Filtered\nNumber of good structure(s): {len(filtered_scores)}\nAll output files: {glob.glob(f"{RELAX_DIR}/good/*.pdb")}\n\nFiltered Score File:\n\n----------\n{filtered_scores.to_csv(index=False)}\n----------\n\nOriginal Score File ({scores_file}):\n\n----------\n{scores.to_csv(index=False)}\n----------\n"
    return f"Fast Relax Designs Filtered\nNumber of good structure(s): {len(filtered_scores)}\nAll output files: {glob.glob(f"{RELAX_DIR}/good/*.pdb")}\n\nScores Distribution Plot: {img_path}\n\nFiltered Score File:\n\n----------\n{filtered_scores.to_csv(index=False)}\n----------\n"


def cleanup():
    # os.system(f"rm -rf /ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/*")
    os.system(f"rm -rf {WDIR}/*")


if __name__ == "__main__":
    mcp.run(transport='stdio')

    # output = separate_ligand_protein_pdb(
    #     pdb_file="/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/inputs/7o2g_HBA.pdb"
    # )
    # print(output)

    # output = generate_cst_file(
    #     res1_atoms=["FE1", "N4", "C19"],
    #     res1_residue="HBA",
    #     res2_atoms=["SH1"],
    #     res2_residue="CYS",
    # )
    # print(output)

    # output = generate_ligand_params_file(
    #     ligand_sdf_file="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output/start/ligand_cleaned.sdf",
    #     ligand="HBA"
    # )
    # print(output)

    # output = convert_ligand_pdb_to_sdf_for_docking(
    #     pdb_path="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output/inputs/HBA.pdb"
    # )
    # print(output)

    # output = run_docking_pipeline(
    #     receptor_path="/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/inputs/protein.pdb",
    #     ligand_path="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output/inputs/ligand_cleaned.sdf",
    #     size_x=40.0,
    #     size_y=40.0,
    #     size_z=40.0,
    #     center_x=-10.0,
    #     center_y=-32.0,
    #     center_z=10.0,
    # )
    # print(output)

    # output = docked_protein_ligand_complex(
    #     receptor_pdbqt_path="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output/inputs/receptor_output.pdbqt",
    #     ligand_pdbqt_path="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output/inputs/docked.pdbqt",
    # )
    # print(output)

    # cleanup()

    # output = run_rf_diffusion(
    #     input_pdb="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb", 
    #     ligand="HBA", 
    #     N_designs=5, 
    #     T_steps=200
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_0.txt", "w") as f:
    #     f.write(output)
    
    # output = analyze_rf_diffusion_outputs(
    #     diffusion_outputs=glob.glob(f"{WDIR}/0_diffusion/7o2g_HBA/out/*.pdb"), 
    #     # ref_pdb="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb", 
    #     ref_pdb="/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/inputs/7o2g_HBA.pdb",
    #     # params=[f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"],
    #     params=["/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output//inputs/HBA.params"],
    #     # term_limit=15.0,
    #     # SASA_limit=0.3,
    #     # loop_limit=0.4,
    #     # longest_helix=30,
    #     # rog=30.0,
    #     ref_catres="A15",
    #     # exclude_clash_atoms="O1 O2 O3 O4 C5 C10",
    #     # ligand_exposed_atoms="C45 C46 C47",
    #     # exposed_atom_SASA=10.0
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_0_analyze.txt", "w") as f:
    #     f.write(output)

    # output = run_protein_mpnn(
    #     diffused_backbones_good=glob.glob(f"{WDIR}/0_diffusion/filtered_structures/*.pdb"),
    #     MPNN_temperatures=[0.1, 0.2, 0.3],
    #     MPNN_outputs_per_temperature=5,
    #     MPNN_omit_AAs="CM",
    #     job_time="1:00:00"
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_1.txt", "w") as f:
    #     f.write(output)

    # output = run_af2(
    #     protein_seq_fasta_files=glob.glob(f"{WDIR}/1_proteinmpnn/seqs/*.fa"),
    #     after_ligand_mpnn=False,
    #     AF2_recycles=3,
    #     AF2_models="4",
    #     job_time="1:00:00"
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_2.txt", "w") as f:
    #     f.write(output)

    # output = analyze_af2_outputs(
    #     params=[f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"],
    #     ref_pdbs_path=f"{WDIR}/0_diffusion/filtered_structures/",
    #     after_ligand_mpnn=False,
    #     af2_output_csv_paths=[f"{WDIR}/2_af2/130aa_0.csv"],
    #     job_time="1:00:00"
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_2_analyze.txt", "w") as f:
    #     f.write(output)

    # output = run_ligand_mpnn(
    #     input_pdbs=glob.glob(f"{WDIR}/2_af2/good/with_heme2/*.pdb"),
    #     # cstfile=f"{SCRIPT_DIR}/theozyme/HBA/HBA_CYS_UPO.cst",
    #     cstfile="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output//start/cst_file.cst",
    #     # params=[f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"],
    #     params=[f"/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/agent_output//inputs/HBA.params"],
    #     # filters={},
    #     # align_atoms=["N1", "N2", "N3", "N4"]
    # )
    # print(output)
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_3.txt", "w") as f:
    #     f.write(output)

    # output = analyze_ligand_mpnn_outputs(
    #     scores_file=f"{WDIR}/3.1_design_pocket_ligandMPNN/scorefile.txt",
    #     filters={
    #         # "all_cst": [1.5, "<="],
    #         "L_SASA": [0.2, "<="],
    #         # "COO_hbond": [1.0, "="],
    #         "cms_per_atom": [5.0, ">="],
    #         "corrected_ddg": [-50.0, "<="],
    #         "nlr_totrms": [0.7, "<="],
    #         # "nlr_SR1_rms": [0.6, "<="]
    #     }
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_3_analyze.txt", "w") as f:
    #     f.write(output)

    # output = run_ligand_mpnn_redesign_2nd_layer_residues(
    #     input_pdbs=glob.glob(f"{WDIR}/3.1_design_pocket_ligandMPNN/good/*.pdb"),
    #     LIGAND="HBA",
    #     params=[f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"],
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_4.txt", "w") as f:
    #     f.write(output)

    # output = run_af2(
    #     protein_seq_fasta_files=glob.glob(f"{WDIR}/4.1_2nd_mpnn/seqs/*.fa"),
    #     after_ligand_mpnn=True,
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_5.txt", "w") as f:
    #     f.write(output)

    # output = analyze_af2_outputs(
    #     params=[f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"],
    #     ref_pdbs_path=f"{WDIR}/3.1_design_pocket_ligandMPNN/good/",
    #     after_ligand_mpnn=True,
    #     lDDT=70.0,
    #     rmsd=20.0,
    #     rmsd_SR1=7.0
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_5_analyze.txt", "w") as f:
    #     f.write(output)

    # output = run_fast_relax(
    #     input_pdbs=glob.glob(f"{WDIR}/5.1_2nd_af2/good/*.pdb"),
    #     ref_pdbs=glob.glob(f"{WDIR}/3.1_design_pocket_ligandMPNN/good/*.pdb"),
    #     cstfile=f"{SCRIPT_DIR}/theozyme/HBA/HBA_CYS_UPO.cst",
    #     LIGAND="HBA",
    #     params=[f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"],
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_6.txt", "w") as f:
    #     f.write(output)

    # output = analyze_fast_relax_outputs(
    #     scores_file=f"{WDIR}/6.1_final_relax/scorefile.txt",
    #     filters={
    #         "all_cst": [1.5, "<="],
    #         "L_SASA": [0.30, "<="],
    #         # "COO_hbond": [1.0, "="],
    #         "cms_per_atom": [3.0, ">="],
    #         # "corrected_ddg": [-50.0, "<="],
    #         "nlr_totrms": [0.8, "<="],
    #         "nlr_SR1_rms": [0.6, "<="]
    #     }
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_6_analyze.txt", "w") as f:
    #     f.write(output)