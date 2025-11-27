import os
from typing import Annotated
from pydantic import Field
import json
import subprocess
import time
import pandas as pd
import numpy as np
import glob
import random
import re

from run_utils import extract_job_id

PPDIFF_PATH = "/ocean/projects/cis240137p/dgarg2/github/PPDiff/"
PPDIFF_CONDA_ENV = "/ocean/projects/cis240137p/dgarg2/miniconda3/envs/ppdiff/bin/python"
COLABFOLD_CACHE = "/ocean/projects/cis240137p/dgarg2/github/colabfold/cf_cache"
COLABFOLD_SIF = "/ocean/projects/cis240137p/dgarg2/github/colabfold/colabfold_1.5.5-cuda12.2.2.sif"
AF3_PATH = "/ocean/projects/cis240137p/dgarg2/github/alphafold3"
AF3_MODELS = "/ocean/projects/cis240137p/dgarg2/af3_models"
AF3_DATA = "/ocean/projects/cis240137p/dgarg2/af3_data"
import sys
sys.path.append(PPDIFF_PATH)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ppdiff")


def parse_pdb_simple(pdb_file, chain="A"):
    seq = []
    coords = []

    three_to_one = {
        'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F',
        'GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L',
        'MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R',
        'SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'
    }

    seen_residues = set()

    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain_id = line[21]
            resnum = int(line[22:26])

            if chain_id != chain:
                continue

            # First time we see this residue, add sequence
            if (chain_id, resnum) not in seen_residues:
                seq.append(three_to_one.get(resname, "X"))
                seen_residues.add((chain_id, resnum))

            # Get CA coords
            if atom_name == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

    return seq, coords


@mcp.tool()
def build_ppdiff_input_binder_design_from_pdb(
    pdb_file: Annotated[str, Field(description="Path to protein PDB")],
    target_length: Annotated[int, Field(description="Length of desired binder")]
) -> str:
    seq, coords = parse_pdb_simple(pdb_file)
    original_len = len(seq)
    seq += ["A"]*target_length
    coords = [[coor]*4 for coor in coords] + [[[0.0, 0.0, 0.0]]*4 for _ in range(target_length)]
    target = [1]*original_len + [0]*target_length
    return build_ppdiff_input_binder_design(seq, ["CA"]*len(target), coords, target)


@mcp.tool()
def build_ppdiff_input_binder_design(
    protein_seq: Annotated[list[str], Field(description="Sequence of the protein")],
    protein_atoms: Annotated[list[str], Field(description="Atoms of the protein")],
    protein_coors: Annotated[list[list[float]], Field(description="Coordinates of the protein")],
    protein_target: Annotated[list[int], Field(description="Generation target of the protein")]
) -> str:
    file_name = os.path.join(PPDIFF_PATH, "data/input_binder_design.json")
    data = {"binder_design": {"test": {"seqs": ["".join(protein_seq)], "atoms": [protein_atoms], "coors": [protein_coors], "target": [protein_target]}}}
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)
    return "Created input file for PPDiff: " + file_name


@mcp.tool()
def build_ppdiff_input_antibody_design_from_pdb(
    pdb_file: Annotated[str, Field(description="Path to protein PDB")],
    cdr_indices: Annotated[list[int], Field(description="Indices of the CDRs")],
    antigen_length: Annotated[int, Field(description="Length of the antigen sequence")],
    heavy_chain_len: Annotated[int, Field(description="Length of the antibody heavy-chain sequence")],
    pdb: Annotated[str, Field(description="4 letter PDB code")]
) -> str:
    seq, coords = parse_pdb_simple(pdb_file)
    target = [1.0]*len(seq)
    for index in cdr_indices:
        target[index] = 0.0
    return build_ppdiff_input_antibody_design(seq, ["CA"]*len(target), coords, target, antigen_length, heavy_chain_len, pdb)


@mcp.tool()
def build_ppdiff_input_antibody_design(
    protein_seq: Annotated[list[str], Field(description="Sequence of the protein")],
    protein_atoms: Annotated[list[str], Field(description="Atoms of the protein")],
    protein_coors: Annotated[list[list[float]], Field(description="Coordinates of the protein")],
    protein_target: Annotated[list[int], Field(description="Generation target of the protein")],
    antigen_length: Annotated[int, Field(description="Length of the antigen sequence")],
    heavy_chain_len: Annotated[int, Field(description="Length of the antibody heavy-chain sequence")],
    pdb: Annotated[str, Field(description="4 letter PDB code")]
) -> str:
    file_name = os.path.join(PPDIFF_PATH, "data/input_antibody_design.json")
    data = {"antibody_design": {"test": {"seqs": ["".join(protein_seq)], "atoms": [protein_atoms], "coors": [protein_coors], "target": [protein_target], "antigen_length": [antigen_length], "heavy_chain_len": [heavy_chain_len], "pdb": [pdb]}}}
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)
    return "Created input file for PPDiff: " + file_name


@mcp.tool()
def run_ppdiff_binder_design(
    input_json: Annotated[str, Field(description="Location of data file")]
) -> str:
    os.chdir(PPDIFF_PATH)

    generation_file = f"""#!/bin/bash\ndata_path={input_json}\nrm -rf {PPDIFF_PATH}/models/output/*\n\nlocal_root=models\noutput_path=${{local_root}}/binder_design\ngeneration_path=${{local_root}}/output/binder_design\nmkdir -p ${{generation_path}}\n\n/ocean/projects/cis240137p/dgarg2/miniconda3/envs/PPDiff/bin/python fairseq_cli/design_binder.py ${{data_path}} --task protein_protein_complex_design --protein-task "binder_design" --dataset-impl "binder_design" --path ${{output_path}}/checkpoint_best.pt --batch-size 1 --results-path ${{generation_path}} --skip-invalid-size-inputs-valid-test --valid-subset test --eval-aa-recovery"""
    with open("design_binder_ppdiff.sh", "w") as f:
        f.write(generation_file)
    os.system("chmod +x design_binder_ppdiff.sh")

    slurm_file = f"""#!/bin/bash\n#SBATCH -N 1\n#SBATCH -p GPU-shared\n#SBATCH -t 1:00:00\n#SBATCH --gpus=v100-32:1\n#SBATCH --output=output.log\n#SBATCH -n 2\n#SBATCH -e output.err\nbash /ocean/projects/cis240137p/dgarg2/github/PPDiff/design_binder_ppdiff.sh"""
    with open("run_slurm.sh", "w") as f:
        f.write(slurm_file)
    os.system("chmod +x run_slurm.sh")

    p = subprocess.Popen(f"cd {PPDIFF_PATH} && sbatch run_slurm.sh", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("PPDiff Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(120)

    with open(f"{PPDIFF_PATH}/output.log", "r") as f:
        logs = f.read()
    with open(f"{PPDIFF_PATH}/output.err", "r") as f:
        errors = f.read()

    return f"PPDiff Finished Successfully\nTarget protein: {PPDIFF_PATH}/models/output/binder_design1/target.txt\nPredicted sequence from PPDiff: {PPDIFF_PATH}/models/output/binder_design1/binder.gen.txt\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


@mcp.tool()
def run_ppdiff_antibody_design(
    input_json: Annotated[str, Field(description="Location of data file")],
    cdr_type: Annotated[int, Field(description="Type of CDR (1, 2, or 3)")],
) -> str:
    os.chdir(PPDIFF_PATH)

    if cdr_type==1:
        cdr = "antibody_design_cdrh1"
    elif cdr_type==2:
        cdr = "antibody_design_cdrh2"
    elif cdr_type==3:
        cdr = "antibody_design_cdrh3"
    else:
        return "Choose a valid CDR type (1, 2, or 3)"

    generation_file = f"""#!/bin/bash\ndata_path={input_json}\nrm -rf {PPDIFF_PATH}/models/output/*\n\nlocal_root=models\noutput_path=${{local_root}}/{cdr}\ngeneration_path=${{local_root}}/output/{cdr}\nmkdir -p ${{generation_path}}\n\n/ocean/projects/cis240137p/dgarg2/miniconda3/envs/PPDiff/bin/python fairseq_cli/design_antibody.py ${{data_path}} --task protein_protein_complex_design --protein-task "antibody_design" --dataset-impl "antibody_design" --path ${{output_path}}/checkpoint_best.pt --batch-size 1 --results-path ${{generation_path}} --skip-invalid-size-inputs-valid-test --valid-subset test --eval-aa-recovery"""
    with open("design_antibody_ppdiff.sh", "w") as f:
        f.write(generation_file)
    os.system("chmod +x design_antibody_ppdiff.sh")

    slurm_file = f"""#!/bin/bash\n#SBATCH -N 1\n#SBATCH -p GPU-shared\n#SBATCH -t 1:00:00\n#SBATCH --gpus=v100-32:1\n#SBATCH --output=output.log\n#SBATCH -n 2\n#SBATCH -e output.err\nbash /ocean/projects/cis240137p/dgarg2/github/PPDiff/design_antibody_ppdiff.sh"""
    with open("run_slurm.sh", "w") as f:
        f.write(slurm_file)
    os.system("chmod +x run_slurm.sh")

    p = subprocess.Popen(f"cd {PPDIFF_PATH} && sbatch run_slurm.sh", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("PPDiff Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(120)

    with open(f"{PPDIFF_PATH}/output.log", "r") as f:
        logs = f.read()
    with open(f"{PPDIFF_PATH}/output.err", "r") as f:
        errors = f.read()

    return f"PPDiff Finished Successfully\nAntigen sequence: {PPDIFF_PATH}/models/output/{cdr}1/antigen.gen.txt\nPredicted heavy chain sequence from PPDiff: {PPDIFF_PATH}/models/output/{cdr}1/heavy.chain.gen.txt\nPredicted light chain sequence from PPDiff: {PPDIFF_PATH}/models/output/{cdr}1/light.chain.gen.txt\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


@mcp.tool()
def run_alphafold3_on_ppdiff_binder_output(
    target_sequence_file: Annotated[str, Field(description="Location of PPDiff target sequence file (target.txt from ppdiff)")],
    binder_sequence_file: Annotated[str, Field(description="Location of PPDiff binder sequence file (binder.gen.txt from ppdiff)")],
    job_time: Annotated[str, Field(description="Time limit for AF2 jobs")] = "6:00:00",
):
    os.chdir(AF3_PATH)
    os.system("mkdir -p inputs/ outputs/")
    
    with open(target_sequence_file, "r") as f:
        target_seq = f.read().strip().replace("\n", "")
    with open(binder_sequence_file, "r") as f:
        binder_seq = f.read().strip().replace("\n", "")
    
    af3_input_json = {
        "name": "ppdiff_protein_binder_complex",
        "dialect": "alphafold3",
        "version": 1,
        "modelSeeds": [1],
        "sequences": [
            {
                "protein": {
                    "id": "T",
                    "sequence": target_seq
                }
            },
            {
                "protein": {
                    "id": "B",
                    "sequence": binder_seq
                }
            }
        ]
    }

    with open("inputs/ppdiff_binder_input.json", "w") as f:
        json.dump(af3_input_json, f, indent=4)

    slurm_file = f"#!/bin/bash\n#SBATCH -N 1\n#SBATCH -n 2\n#SBATCH -p GPU-shared\n#SBATCH -t 24:00:00\n#SBATCH --gpus=h100-80:1\n#SBATCH --output=output.log\n#SBATCH -e output.err\n\napptainer exec --nv --bind {AF3_PATH}/inputs:/root/af_input --bind {AF3_PATH}/outputs:/root/af_output --bind {AF3_MODELS}:/root/models --bind {AF3_DATA}:/root/public_databases alphafold3.sif python /app/alphafold/run_alphafold.py --json_path=/root/af_input/ppdiff_binder_input.json --model_dir=/root/models --db_dir=/root/public_databases --output_dir=/root/af_output\n"
    with open("run_slurm.sh", "w") as f:
        f.write(slurm_file)
    
    p = subprocess.Popen(f"cd {AF3_PATH} && sbatch run_slurm.sh", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("PPDiff Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(300)

    with open("outputs/ppdiff_protein_binder_complex/ppdiff_protein_binder_complex_confidences.json") as f:
        file = json.load(f)
        pae = np.mean(file["pae"])
        plddt = np.mean(file["atom_plddts"])
    with open("outputs/ppdiff_protein_binder_complex/ppdiff_protein_binder_complex_summary_confidences.json") as f:
        file = json.load(f)
        iptm = file["iptm"]
        ptm = file["ptm"]
    complex_file = "outputs/ppdiff_protein_binder_complex/ppdiff_protein_binder_complex_model.cif"
    with open("output.log", "r") as f:
        logs = f.read()
    with open("output.err", "r") as f:
        errors = f.read()

    return f"AlphaFold3 Finished Successfully\nPredicted complex structure: {complex_file}\npLDDT: {plddt}\nipTM: {iptm}\npTM: {ptm}\nPAE: {pae}\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


@mcp.tool()
def run_alphafold3_on_antibody_binder_output(
    antigen_sequence_file: Annotated[str, Field(description="Location of PPDiff target sequence file (antigen.gen.txt from ppdiff)")],
    heavy_chain_sequence_file: Annotated[str, Field(description="Location of PPDiff binder sequence file (heavy.chain.gen.txt from ppdiff)")],
    light_chain_sequence_file: Annotated[str, Field(description="Location of PPDiff binder sequence file (light.chain.gen.txt from ppdiff)")],
    job_time: Annotated[str, Field(description="Time limit for AF2 jobs")] = "6:00:00",
):
    os.chdir(AF3_PATH)
    os.system("mkdir -p inputs/ outputs/")
    
    with open(antigen_sequence_file, "r") as f:
        antigen_seq = f.read().strip().replace("\n", "")
    with open(heavy_chain_sequence_file, "r") as f:
        heavy_chain_seq = f.read().strip().replace("\n", "")
    with open(light_chain_sequence_file, "r") as f:
        light_chain_seq = f.read().strip().replace("\n", "")
    
    af3_input_json = {
        "name": "ppdiff_antibody_antigen_complex",
        "dialect": "alphafold3",
        "version": 1,
        "modelSeeds": [1],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": antigen_seq
                }
            },
            {
                "protein": {
                    "id": "H",
                    "sequence": heavy_chain_seq
                }
            },
            {
                "protein": {
                    "id": "L",
                    "sequence": light_chain_seq
                }
            }
        ]
    }

    with open("inputs/ppdiff_antibody_input.json", "w") as f:
        json.dump(af3_input_json, f, indent=4)

    slurm_file = f"#!/bin/bash\n#SBATCH -N 1\n#SBATCH -n 2\n#SBATCH -p GPU-shared\n#SBATCH -t 24:00:00\n#SBATCH --gpus=h100-80:1\n#SBATCH --output=output.log\n#SBATCH -e output.err\n\napptainer exec --nv --bind {AF3_PATH}/inputs:/root/af_input --bind {AF3_PATH}/outputs:/root/af_output --bind {AF3_MODELS}:/root/models --bind {AF3_DATA}:/root/public_databases alphafold3.sif python /app/alphafold/run_alphafold.py --json_path=/root/af_input/ppdiff_antibody_input.json --model_dir=/root/models --db_dir=/root/public_databases --output_dir=/root/af_output\n"
    with open("run_slurm.sh", "w") as f:
        f.write(slurm_file)
    
    p = subprocess.Popen(f"cd {AF3_PATH} && sbatch run_slurm.sh", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("PPDiff Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(300)

    with open("outputs/ppdiff_antibody_antigen_complex/ppdiff_antibody_antigen_complex_confidences.json") as f:
        file = json.load(f)
        pae = np.mean(file["pae"])
        plddt = np.mean(file["atom_plddts"])
    with open("outputs/ppdiff_antibody_antigen_complex/ppdiff_antibody_antigen_complex_summary_confidences.json") as f:
        file = json.load(f)
        iptm = file["iptm"]
        ptm = file["ptm"]
    complex_file = "outputs/ppdiff_antibody_antigen_complex/ppdiff_antibody_antigen_complex_model.cif"
    with open("output.log", "r") as f:
        logs = f.read()
    with open("output.err", "r") as f:
        errors = f.read()

    return f"AlphaFold3 Finished Successfully\nPredicted complex structure: {complex_file}\npLDDT: {plddt}\nipTM: {iptm}\npTM: {ptm}\nPAE: {pae}\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


if __name__ == "__main__":
    mcp.run(transport='stdio')

    # out = build_ppdiff_input_binder_design_from_pdb(
    #     pdb_file="/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/inputs/binder_design_example.pdb",
    #     target_length=65
    # )
    # print(out)

    # time.sleep(5)
    # out = run_ppdiff_binder_design("/ocean/projects/cis240137p/dgarg2/github/PPDiff/data/input_binder_design.json")
    # print(out)

    # out = build_ppdiff_input_antibody_design_from_pdb(
    #     pdb_file="/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/inputs/antibody_design_example_1.pdb",
    #     cdr_indices=[193, 194, 195, 196, 197, 198, 199, 200, 218, 219, 220, 221, 222, 223, 224, 225, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 316, 317, 318, 319, 320, 321, 322, 340, 341, 342, 379, 380, 381, 382, 383, 384, 385, 386],
    #     antigen_length=168,
    #     heavy_chain_len=124,
    #     pdb="8tzy"
    # )
    # print(out)

    # time.sleep(5)
    # out = run_ppdiff_antibody_design("/ocean/projects/cis240137p/dgarg2/github/PPDiff/data/input_antibody_design.json", 1)
    # print(out)

    # out = run_alphafold3_on_ppdiff_binder_output(
    #     "/ocean/projects/cis240137p/dgarg2/github/PPDiff/models/output/binder_design1/target.txt",
    #     "/ocean/projects/cis240137p/dgarg2/github/PPDiff/models/output/binder_design1/binder.gen.txt"
    # )
    # print(out)

    # out = run_alphafold3_on_antibody_binder_output(
    #     "/ocean/projects/cis240137p/dgarg2/github/PPDiff/models/output/antibody_design_cdrh11/antigen.gen.txt",
    #     "/ocean/projects/cis240137p/dgarg2/github/PPDiff/models/output/antibody_design_cdrh11/heavy.chain.gen.txt",
    #     "/ocean/projects/cis240137p/dgarg2/github/PPDiff/models/output/antibody_design_cdrh11/light.chain.gen.txt"
    # )
    # print(out)
