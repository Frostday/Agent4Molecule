import os
from typing import Annotated
from pydantic import Field
import json
import subprocess
import time
import pandas as pd
import numpy as np
import glob

from run_utils import extract_job_id

ENZYGEN_PATH = "/ocean/projects/cis240137p/dgarg2/github/EnzyGen/"
ENZYGEN_CONDA_ENV = "/ocean/projects/cis240137p/dgarg2/miniconda3/envs/enzygen/bin/python"
import sys
sys.path.append(ENZYGEN_PATH)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("enzygen")


@mcp.tool()
def find_enzyme_category_using_keywords(
    keywords: Annotated[list[str], Field(description="Keywords containing the enzyme related information like chemical compound names, gene names etc.")],
) -> str:
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
    substrate_file: Annotated[str, Field(description="Substrate file")] = None,
) -> str:
    file_name = os.path.join(ENZYGEN_PATH, "data/input.json")
    data = {}
    indices = ",".join([str(i) for i in sorted(motif_indices)])+"\n"
    pdb, ec4, substrate = ref_pdb_chain, enzyme_family, substrate_file
    seq, coord = "", ""
    idx = 0
    for i in range(recommended_length):
        if i in motif_indices:
            idx = motif_indices.index(i)
            seq += motif_seq[idx]
            coord += ",".join([str(i) for i in motif_coord[idx]])+","
        else:
            seq += "A"
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
    if substrate:
        data[".".join(enzyme_family.split(".")[:-1])]["test"]["substrate"] = [substrate]
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
    return f"EnzyGen Finished Successfully\nPredicted structure(s) from EnzyGen: {glob.glob(f"{ENZYGEN_PATH}/outputs/*/pred_pdbs/*.pdb")}\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


def cleanup():
    os.system(f"rm -rf {ENZYGEN_PATH}/outputs/*")
    os.system(f"rm -f {ENZYGEN_PATH}/run_enzygen.sh")
    os.system(f"rm -f {ENZYGEN_PATH}/run_gpu_slurm.sh")
    os.system(f"rm -f {ENZYGEN_PATH}/data/input.json")
    os.system(f"rm -f {ENZYGEN_PATH}/output.log")
    os.system(f"rm -f {ENZYGEN_PATH}/output.err")


if __name__ == "__main__":
    mcp.run(transport='stdio')
    # print(find_enzyme_category_using_keywords(["oxidase", "D-ARABINONO-1,4-LACTONE"]))
    # print(find_enzyme_category_using_keywords(["adenylate", "cyclase", "adenylylcyclase"]))
    # print(find_mined_motifs_enzyme_category("4.6.1.1", start_index=0, end_index=2))
    # cleanup()
    # print(build_enzygen_input("4.6.1.1", [7, 9, 13, 16, 19, 20, 22, 28, 58, 59, 63, 67, 75, 82, 84, 90, 115, 117, 119, 123, 130, 138, 143, 149, 154, 158, 167], ['R', 'I', 'F', 'I', 'F', 'T', 'M', 'S', 'G', 'D', 'A', 'A', 'E', 'A', 'A', 'A', 'R', 'G', 'H', 'A', 'S', 'A', 'V', 'L', 'A', 'I', 'Y'], [[43.506, -6.758, 46.718], [38.345, -4.765, 44.212], [27.09, 0.648, 48.385], [17.123, 3.522, 50.054], [11.402, 2.391, 54.192], [10.907, 3.571, 57.767], [6.486, 3.519, 54.572], [5.508, -5.321, 61.885], [19.634, -1.77, 57.845], [18.156, -1.1, 54.433], [28.436, -5.703, 47.564], [36.463, -11.125, 46.134], [34.608, -7.277, 34.653], [24.991, -3.646, 35.749], [23.48, -0.824, 40.253], [13.852, -3.122, 40.484], [21.286, 5.9, 46.057], [28.037, 4.6, 45.147], [34.503, 2.711, 43.701], [41.431, -2.971, 49.986], [38.431, -12.864, 66.38], [36.711, -5.303, 52.874], [35.782, 1.19, 51.082], [30.113, 8.463, 47.15], [21.426, 12.621, 44.494], [24.48, 6.126, 40.615], [35.757, 1.433, 32.015]], 198, "1wc3.A", substrate_file="CHEBI_57540.sdf"))
    # print(build_enzygen_input("4.6.1.1", [0, 1, 4], ['D', 'I', 'G'], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], 5, substrate_file="CHEBI_57540.sdf"))
    # print(run_enzygen(f"{ENZYGEN_PATH}/data/input.json"))
    # print(run_enzygen(f"{ENZYGEN_PATH}/data/test_2.json"))
