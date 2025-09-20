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

from run_utils import extract_job_id

ENZYGEN_PATH = "/ocean/projects/cis240137p/dgarg2/github/EnzyGen/"
ENZYGEN_CONDA_ENV = "/ocean/projects/cis240137p/dgarg2/miniconda3/envs/enzygen/bin/python"
import sys
sys.path.append(ENZYGEN_PATH)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("enzygen")

# AF2 setup
CONDAPATH = "/ocean/projects/cis240137p/dgarg2/miniconda3"
HEME_BINDER_PATH = "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/"
SCRIPT_DIR = os.path.dirname(HEME_BINDER_PATH)
AF2_script = f"{SCRIPT_DIR}/scripts/af2/af2.py"
PYTHON = {
    "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    "general": f"{CONDAPATH}/envs/diffusion/bin/python"
}
sys.path.append(HEME_BINDER_PATH)
sys.path.append(SCRIPT_DIR+"/scripts/utils")
import utils


@mcp.tool()
def find_enzyme_category_using_keywords(
    keywords: Annotated[list[str], Field(description="Keywords containing the enzyme related information like chemical compound names, gene names etc.")],
) -> str:
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)
    df = pd.read_csv("data/enzymes.txt", sep="\t", header=None, names=["EC4", "description"])
    matches = [df['description'].str.contains(kw, case=False) for kw in keywords]
    match_counts = pd.concat(matches, axis=1).sum(axis=1)
    df['match_count'] = match_counts
    result = df[df['match_count'] > 0].sort_values(by='match_count', ascending=False)
    return "Top 5 results:\n\n" + result.head(5).to_csv(index=False)


@mcp.tool()
def find_mined_motifs_enzyme_category(
    enzyme_category: Annotated[str, Field(description="Enzyme category (e.g. 4.6.1.1)")],
    start_index: Annotated[int, Field(description="Start index (suggestion: only extract 2-5 at a time)")] = 0,
    end_index: Annotated[int, Field(description="End index (suggestion: only extract 2-5 at a time)")] = 5,
) -> str:
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)
    if end_index < start_index:
        return "End index should be greater than start index"
    with open("data/mined_motifs.json") as f:
        data = json.load(f)
    if enzyme_category not in data.keys():
        return "No motif information available for the enzyme category - try another EC4 category or ask the user for motif information"
    data = data[enzyme_category]
    options = []
    if end_index > len(data):
        return "Number of mined motifs is less than the end index (total: {})".format(len(data))
    for i, d in enumerate(data[start_index:end_index]):
        text = f"- Test {i+1}\n\t- Motif Indices: {d['motif']}\n\t- Motif Sequence: {np.array(d['seq'])[d['motif']].tolist()}\n\t- Motif coordinates: {np.array(d['coor'])[d['motif']].tolist()}\n\t- Recommended length: {len(d['seq'])}\n\t- Reference PDB: {d['pdb']}"
        options.append(text)
    return f"Total motif options: {len(data)}\n\nHere are some mined motifs for the enzyme family {enzyme_category}:\n\n" + "\n".join(options)


@mcp.tool()
def build_enzygen_input(
    enzyme_family: Annotated[str, Field(description="Enzyme family of the enzyme to be generated (EC4 category e.g. 1.1.1.1)")],
    motif_indices: Annotated[list[int], Field(description="Indices of the motif")],
    motif_seq: Annotated[list[str], Field(description="Sequence of the motif")],
    motif_coord: Annotated[list[list[float]], Field(description="Coordinates of the motif")],
    recommended_length: Annotated[int, Field(description="Recommended length of the enzyme to be generated")],
    ref_pdb_chain: Annotated[str, Field(description="Reference PDB chain")] = "AAAA.A",
    # substrate_file: Annotated[str, Field(description="Substrate file")] = None,
) -> str:
    file_name = os.path.join(ENZYGEN_PATH, "data/input.json")
    data = {}
    indices = ",".join([str(i) for i in sorted(motif_indices)])+"\n"
    pdb, ec4 = ref_pdb_chain, enzyme_family
    seq, coord = "", ""
    idx = 0
    amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    for i in range(recommended_length):
        if i in motif_indices:
            idx = motif_indices.index(i)
            seq += motif_seq[idx]
            coord += ",".join([str(i) for i in motif_coord[idx]])+","
        else:
            seq += "A"
            # seq += random.choice(amino_acids)
            coord += "0.0,0.0,0.0,"
    coord = coord[:-1]
    data = {
        ".".join(enzyme_family.split(".")[:-1]): {
            "test": {
                "seq": [seq],
                "coor": [coord],
                "motif": [indices],
                "pdb": [pdb],
                "ec4": [ec4],
            }
        }
    }
    # if substrate_file:
    #     data[".".join(enzyme_family.split(".")[:-1])]["test"]["substrate"] = [substrate_file]
    with open(file_name, 'w') as f:
        f.write(json.dumps(data, indent=4))
    return "Created input file for Enzygen: " + file_name


@mcp.tool()
def run_enzygen(input_json: Annotated[str, Field(description="Location of script directory")]) -> str:
    with open(input_json, "r") as f:
        input_data = json.load(f)
    enzymes_families = input_data.keys()
    text = f"""#!/bin/bash\n\nrm -rf outputs/*\n\ndata_path={input_json}\n\noutput_path=models\nproteins=({" ".join(enzymes_families)})\n\nfor element in ${{proteins[@]}}\ndo\ngeneration_path={ENZYGEN_PATH}/outputs/${{element}}\n\nmkdir -p ${{generation_path}}\nmkdir -p ${{generation_path}}/pred_pdbs\nmkdir -p ${{generation_path}}/tgt_pdbs\n\n{ENZYGEN_CONDA_ENV} fairseq_cli/validate.py ${{data_path}} --task geometric_protein_design --protein-task ${{element}} --dataset-impl-source "raw" --dataset-impl-target "coor" --path ${{output_path}}/checkpoint_best.pt --batch-size 1 --results-path ${{generation_path}} --skip-invalid-size-inputs-valid-test --valid-subset test --eval-aa-recovery\ndone"""
    run_file = os.path.join(ENZYGEN_PATH, "run_enzygen.sh")
    slurm_file = os.path.join(ENZYGEN_PATH, "run_gpu_slurm.sh")
    with open(run_file, "w") as f:
        f.write(text)
    with open(slurm_file, "w") as f:
        f.write(f"#!/bin/bash\n#SBATCH -N 1\n#SBATCH -p GPU-small\n#SBATCH -t 1:00:00\n#SBATCH --gpus=v100-32:1\n#SBATCH --output=output.log\n#SBATCH -n 2\n#SBATCH -e output.err\nbash {run_file}")
    os.system(f"chmod +x {run_file}")
    os.system(f"chmod +x {slurm_file}")
    
    p = subprocess.Popen(f"cd {ENZYGEN_PATH} && sbatch {slurm_file}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("EnzyGen Job ID:", job_id)
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("Job still running...")
        time.sleep(60)

    output_preds = []
    for enzyme_family in os.listdir(f"{ENZYGEN_PATH}/outputs"):
        for output in os.listdir(f"{ENZYGEN_PATH}/outputs/{enzyme_family}/pred_pdbs"):
            if output.endswith(".pdb"):
                with open(f"{ENZYGEN_PATH}/outputs/{enzyme_family}/pred_pdbs/{output}", "r") as f:
                    content = f.read()
                output_preds.append(enzyme_family + "\n" + output + "\n\n" + content)
    with open(f"{ENZYGEN_PATH}/output.log", "r") as f:
        logs = f.read()
    with open(f"{ENZYGEN_PATH}/output.err", "r") as f:
        errors = f.read()
    
    # return "Predicted structure(s) from EnzyGen:\n\n----------\n" + "\n----------\n----------\n".join(output_preds) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"
    return f"EnzyGen Finished Successfully\nPredicted sequence from EnzyGen: {glob.glob(f"{ENZYGEN_PATH}/outputs/*/protein.txt")[0]}\nPredicted structure from EnzyGen: {glob.glob(f"{ENZYGEN_PATH}/outputs/*/pred_pdbs/*.pdb")[0]}\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


@mcp.tool()
def run_af2_on_enzygen_output(
    sequence_file: Annotated[str, Field(description="Location of enzygen sequence file (protein.txt from enzygen)")],
    AF2_recycles: Annotated[int, Field(description="Number of AF2 recycling steps")] = 3,
    AF2_models: Annotated[str, Field(description="Number of AF2 models (default is '4', add other models to this string if needed, i.e. '3 4 5')")] = "4",
    job_time: Annotated[str, Field(description="Time limit for AF2 jobs")] = "1:00:00",
):
    AF2_DIR = f"{ENZYGEN_PATH}/af2_outputs/"
    os.makedirs(AF2_DIR, exist_ok=True)
    os.chdir(AF2_DIR)
    os.system(f"rm -rf {AF2_DIR}/*")
    
    with open(sequence_file, "r") as f:
        sequence = f.read()
    with open("input_seq.fasta", "w") as f:
        f.write(">enzygen\n" + sequence)
    
    cmds_filename_af2 = "commands_af2"
    with open(cmds_filename_af2, "w") as file:
        command = f"{PYTHON['af2']} {AF2_script} --af-nrecycles {AF2_recycles} --af-models {AF2_models} --fasta input_seq.fasta --scorefile scores.csv\n"
        file.write(command)

    submit_script = "submit_af2.sh"
    utils.create_slurm_submit_script(
        filename=submit_script, 
        N_cores=2, 
        time=job_time, 
        array=1,
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
    with open(glob.glob(f"{AF2_DIR}/*.csv")[0], "r") as f:
        af2_scores_csv = f.read()
    af2_out_files = glob.glob(f"{AF2_DIR}/*.pdb")

    return f"AF2 Job Completed\nAll output files: {af2_out_files}\n\nAF2 Scores CSVs: {glob.glob(f"{AF2_DIR}/*.csv")}\n\n----------\n" + af2_scores_csv + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


def cleanup():
    os.system(f"rm -rf {ENZYGEN_PATH}/outputs/*")
    os.system(f"rm -rf {ENZYGEN_PATH}/af2_outputs/*")
    os.system(f"rm -f {ENZYGEN_PATH}/run_enzygen.sh")
    os.system(f"rm -f {ENZYGEN_PATH}/run_gpu_slurm.sh")
    os.system(f"rm -f {ENZYGEN_PATH}/data/input.json")
    os.system(f"rm -f {ENZYGEN_PATH}/output.log")
    os.system(f"rm -f {ENZYGEN_PATH}/output.err")


if __name__ == "__main__":
    mcp.run(transport='stdio')
    # print(find_enzyme_category_using_keywords(["oxidase", "D-ARABINONO-1,4-LACTONE"]))
    # print(find_enzyme_category_using_keywords(["adenylate", "cyclase", "adenylylcyclase"]))
    # print(find_mined_motifs_enzyme_category("4.6.1.1", start_index=0, end_index=1))
    # cleanup()
    # print(build_enzygen_input("4.6.1.1", [7, 9, 13, 16, 19, 20, 22, 28, 58, 59, 63, 67, 75, 82, 84, 90, 115, 117, 119, 123, 130, 138, 143, 149, 154, 158, 167], ['R', 'I', 'F', 'I', 'F', 'T', 'M', 'S', 'G', 'D', 'A', 'A', 'E', 'A', 'A', 'A', 'R', 'G', 'H', 'A', 'S', 'A', 'V', 'L', 'A', 'I', 'Y'], [[43.506, -6.758, 46.718], [38.345, -4.765, 44.212], [27.09, 0.648, 48.385], [17.123, 3.522, 50.054], [11.402, 2.391, 54.192], [10.907, 3.571, 57.767], [6.486, 3.519, 54.572], [5.508, -5.321, 61.885], [19.634, -1.77, 57.845], [18.156, -1.1, 54.433], [28.436, -5.703, 47.564], [36.463, -11.125, 46.134], [34.608, -7.277, 34.653], [24.991, -3.646, 35.749], [23.48, -0.824, 40.253], [13.852, -3.122, 40.484], [21.286, 5.9, 46.057], [28.037, 4.6, 45.147], [34.503, 2.711, 43.701], [41.431, -2.971, 49.986], [38.431, -12.864, 66.38], [36.711, -5.303, 52.874], [35.782, 1.19, 51.082], [30.113, 8.463, 47.15], [21.426, 12.621, 44.494], [24.48, 6.126, 40.615], [35.757, 1.433, 32.015]], 198, "1wc3.A"))
    # print(build_enzygen_input("4.6.1.1", [0, 1, 4], ['D', 'I', 'G'], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], 5))
    # print(run_enzygen(f"{ENZYGEN_PATH}/data/input.json"))
    # print(run_enzygen(f"{ENZYGEN_PATH}/data/test_2.json"))
    # print(run_af2_on_enzygen_output("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/outputs/4.6.1/protein.txt"))

    # print(find_mined_motifs_enzyme_category("3.5.2.6", start_index=0, end_index=1))
    # print(build_enzygen_input(
    #     "3.5.2.6", 
    #     [22, 42, 43, 62, 81, 88, 92, 101, 120, 144, 155, 161, 167, 169, 170, 171, 226, 230, 232, 256, 275],
    #     ['K', 'T', 'A', 'F', 'E', 'L', 'V', 'V', 'L', 'D', 'S', 'F', 'A', 'T', 'L', 'N', 'I', 'L', 'K', 'I', 'Q'],
    #     [[22.755, 2.935, 10.011], [14.361, -21.568, -2.35], [10.566, -21.912, -1.831], [33.479, -7.388, 0.469], [34.619, -36.931, -8.252], [40.085, -37.687, -0.118], [49.359, -34.678, 3.476], [50.436, -19.875, 10.878], [41.992, -31.006, 6.005], [44.234, -18.75, -8.763], [33.671, -23.211, -13.436], [34.724, -14.564, -6.591], [45.993, -17.545, 4.938], [46.929, -12.566, 4.584], [43.669, -13.257, 2.745], [41.943, -11.729, 5.837], [23.717, -16.24, 11.774], [17.384, -18.056, 14.149], [17.479, -22.741, 18.095], [19.302, -22.921, 6.657], [41.661, -1.35, 10.927]],
    #     298,
    #     "2zqc.A"
    # ))
    # print(run_enzygen(f"{ENZYGEN_PATH}/data/input.json"))
    # print(run_af2_on_enzygen_output("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/outputs/3.5.2/protein.txt"))
