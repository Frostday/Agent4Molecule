import os
from typing import Annotated
from pydantic import Field
import json
import subprocess
import time

ENZYGEN_PATH = "/ocean/projects/cis240137p/dgarg2/github/EnzyGen/"
ENZYGEN_CONDA_ENV = "/ocean/projects/cis240137p/dgarg2/miniconda3/envs/enzygen/bin/python"
import sys
sys.path.append(ENZYGEN_PATH)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("enzygen")


def extract_job_id(output: str) -> str:
    """Extracts the job ID from the output of the sbatch command."""
    lines = output.split('\n')
    for line in lines:
        if "Submitted batch job" in line:
            return line.split()[-1]
    return ""


@mcp.tool()
def build_enzygen_input(
    enzyme_family: Annotated[str, Field(description="Enzyme family name")],
    motif_seq: Annotated[str, Field(description="Sequence of the motif")],
    motif_coord: Annotated[list[int], Field(description="Coordinates of the motif")],
    motif_indices: Annotated[list[int], Field(description="Indices of the motif")],
    motif_pdb: Annotated[str, Field(description="PDB file of the motif")],
    motif_ec4: Annotated[str, Field(description="EC4 file of the motif")],
    motif_substrate: Annotated[str, Field(description="Substrate file of the motif")],
    recommended_length: Annotated[int, Field(description="Recommended length of the motif")]
) -> str:
    file_name = ENZYGEN_PATH+"/data/input.json"
    data = {}
    indices, pdb, ec4, substrate = ",".join([str(i) for i in motif_indices])+"\n", motif_pdb, motif_ec4, motif_substrate
    seq, coord = "", ""
    idx = 0
    for i in range(recommended_length):
        if i in motif_indices:
            seq += motif_seq[idx]
            coord += ",".join([str(i) for i in motif_coord[idx*3:idx*3+3]])+","
            idx += 1
        else:
            seq += "A"
            coord += "0.0,0.0,0.0,"
    coord = coord[:-1]
    data = {
        enzyme_family: {
            "test": {
                "seq": [seq],
                "coor": [coord],
                "motif": [indices],
                "pdb": [pdb],
                "ec4": [ec4],
                "substrate": [substrate]
            }
        }
    }
    with open(file_name, 'w') as f:
        f.write(json.dumps(data, indent=4))
    return "Created input file for Enzygen: " + file_name


@mcp.tool()
def run_enzygen(input_json: Annotated[str, Field(description="Location of script directory")]) -> str:
    with open(input_json, "r") as f:
        input_data = json.load(f)
    enzymes_families = input_data.keys()
    text = f"""#!/bin/bash\n\nrm -rf outputs/*\n\ndata_path={input_json}\n\noutput_path=models\nproteins=({" ".join(enzymes_families)})\n\nfor element in ${{proteins[@]}}\ndo\ngeneration_path={ENZYGEN_PATH}/outputs/${{element}}\n\nmkdir -p ${{generation_path}}\nmkdir -p ${{generation_path}}/pred_pdbs\nmkdir -p ${{generation_path}}/tgt_pdbs\n\n{ENZYGEN_CONDA_ENV} fairseq_cli/validate.py ${{data_path}} --task geometric_protein_design --protein-task ${{element}} --dataset-impl-source "raw" --dataset-impl-target "coor" --path ${{output_path}}/checkpoint_best.pt --batch-size 1 --results-path ${{generation_path}} --skip-invalid-size-inputs-valid-test --valid-subset test --eval-aa-recovery\ndone"""
    run_file = ENZYGEN_PATH+"/run_enzygen.sh"
    slurm_file = ENZYGEN_PATH+"/run_gpu_slurm.sh"
    with open(run_file, "w") as f:
        f.write(text)
    with open(slurm_file, "w") as f:
        f.write(f"#!/bin/bash\n#SBATCH -N 1\n#SBATCH -p GPU-shared\n#SBATCH -t 1:00:00\n#SBATCH --gpus=v100-32:1\n#SBATCH --output=output.log\n#SBATCH -n 2\n#SBATCH -e output.err\n#SBATCH -a 1-1\nbash {run_file}")
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
        time.sleep(300)

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
    return "Predicted structure(s) from EnzyGen:\n\n" + "\n----------\n".join(output_preds) + "\n----------\n" + "Log File:\n\n" + logs + "\n----------\n" + "Error File:\n\n" + errors


def cleanup():
    os.system(f"rm -rf {ENZYGEN_PATH}/outputs/*")
    os.system(f"rm -f {ENZYGEN_PATH}/run_enzygen.sh")
    os.system(f"rm -f {ENZYGEN_PATH}/run_gpu_slurm.sh")
    os.system(f"rm -f {ENZYGEN_PATH}/data/input.json")
    os.system(f"rm -f {ENZYGEN_PATH}/output.log")
    os.system(f"rm -f {ENZYGEN_PATH}/output.err")


if __name__ == "__main__":
    mcp.run(transport='stdio')
    # cleanup()
    # print(get_motif_sequence("4.6.1", "DIG", [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], [0, 1, 4], "5cxl.A", "4.6.1.1", "CHEBI_57540.sdf", 5))
    # print(run_enzygen(f"{ENZYGEN_PATH}/data/input.json"))
    # print(run_enzygen(f"{ENZYGEN_PATH}/data/test_2.json"))
