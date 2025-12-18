import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime


import numpy as np
import pandas as pd
from pydantic import Field
from rdkit import Chem
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from run_utils import extract_job_id

# ==================== CONFIGURATION ====================
# IMPORTANT: Configure these paths for your environment before running
# See ENZYGEN_SETUP.md for detailed setup instructions

# Path to EnzyGen repository
ENZYGEN_PATH = "/path/to/EnzyGen"

# Path to EnzyGen conda environment Python executable
ENZYGEN_CONDA_ENV = "/path/to/anaconda3/envs/enzygen/bin/python"

# ColabFold paths
COLABFOLD_CACHE = "/path/to/colabfold/cf_cache"
COLABFOLD_SIF = "/path/to/colabfold/colabfold_1.5.5-cuda12.2.2.sif"

# Utility script path
combine_protein_ligand_file = "/path/to/Agent4Molecule/mcp_agent/util/combine_protein_ligand.py"

# Python environments for different tools
PYTHON = {
    "diffusion": "/path/to/anaconda3/envs/diffusion/bin/python",
    "vina": "/path/to/anaconda3/envs/docking/bin/python"
}

# Conda environment names
DOCKING_ENV_NAME = "docking"
ESP_CONDA_ENV = "esp"
FASTMD_CONDA_ENV = "fastmds"

# Paths to other repositories
FASTMD_PATH = "/path/to/FastMDSimulation"
MOLECULE_AGENT_PATH = "/path/to/Agent4Molecule/"

# Working directory (can be customized per run)
EC_FOLDER = "2.4.1.135"

# =======================================================
sys.path.append(ENZYGEN_PATH)

three_to_one = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "SEC":"U","PYL":"O","ASX":"B","GLX":"Z","XAA":"X","UNK":"X"
}


mcp = FastMCP("enzygen")


@mcp.tool()
def find_enzyme_category_using_keywords(
    keywords: Annotated[list[str], Field(description="Keywords containing the enzyme related information like chemical compound names, gene names etc.")],
) -> dict:
    """
    Find enzyme categories using keywords.
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)
    df = pd.read_csv("data/enzymes.txt", sep="\t", header=None, names=["EC4", "description"])
    matches = [df['description'].str.contains(kw, case=False) for kw in keywords]
    match_counts = pd.concat(matches, axis=1).sum(axis=1)
    df['match_count'] = match_counts
    result = df[df['match_count'] > 0].sort_values(by='match_count', ascending=False)

    answer_dict = {"status": "successful"}
    answer_dict["answer"] = "Top 5 results:\n\n" + result.head(5).to_csv(index=False)
    answer_dict["visualize"] = "none"
    answer_dict["message_render"] = "table"
    return answer_dict


@mcp.tool()
def find_mined_motifs_enzyme_category(
    enzyme_category: Annotated[str, Field(description="Enzyme category (e.g. 4.6.1.1)")],
    start_index: Annotated[int, Field(description="Start index is inclusive (suggestion: only extract 2-5 at a time)")] = 0,
    end_index: Annotated[int, Field(description="End index is exclusive (suggestion: only extract 2-5 at a time)")] = 2,
) -> dict:
    """ 
    Find mined motifs for a given enzyme category.
    """
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
        text = f"- Test {i+1}\n\t- Motif indices: {d['motif']}\n\t- Motif sequence: {np.array(d['seq'])[d['motif']].tolist()}\n\t- Motif coordinates: {np.array(d['coor'])[d['motif']].tolist()}\n\t- Recommended length: {len(d['seq'])}\n\t- Reference PDB: {d['pdb']}"
        options.append(text)

    answer_dict = {"status": "successful"}
    answer_dict['answer'] = f"Total motif options: {len(data)}\n\nHere are some mined motifs for the enzyme family {enzyme_category}:\n\n" + "\n".join(options)
    answer_dict['visualize'] = "none"
    answer_dict["message_render"] = 'text'
    return answer_dict


@mcp.tool()
def build_enzygen_input(
    output_dir: Annotated[str,Field(description="Directory path to store enzygen input JSON")],
    enzyme_family: Annotated[str, Field(description="Enzyme family of the enzyme to be generated (EC4 category e.g. 1.1.1.1)")],
    motif_indices: Annotated[list[int], Field(description="Indices of the motif")],
    motif_seq: Annotated[list[str], Field(description="Sequence of the motif")],
    motif_coord: Annotated[list[list[float]], Field(description="Coordinates of the motif")],
    recommended_length: Annotated[int, Field(description="Recommended length of the enzyme to be generated")],
    ref_pdb_chain: Annotated[str, Field(description="Reference PDB chain")] = "AAAA.A",
) -> dict:
    file_name = os.path.join(ENZYGEN_PATH, "data/input.json")
    dt = f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
    data = {}
    indices = ",".join([str(i) for i in sorted(motif_indices)])+"\n"
    pdb, ec4 = ref_pdb_chain, enzyme_family

    seq, coord = "", ""
    amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    for i in range(recommended_length):
        if i in motif_indices:
            idx = motif_indices.index(i)
            seq += motif_seq[idx]
            coord += ",".join([str(i) for i in motif_coord[idx]])+","
        else:
            seq += "A"
            coord += "0.0,0.0,0.0,"
    if coord[-1] == ",":
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
    with open(file_name, 'w') as f:
        f.write(json.dumps(data, indent=4))

    answer_dict = {"status": "successful"}
    answer_dict["answer"] = "Created input file for Enzygen: " + file_name
    answer_dict["visualize"] = "none"
    answer_dict["message_render"] = "text"
    answer_dict["file_path"] = file_name
    return answer_dict


@mcp.tool()
def change_specific_residues_using_enzygen_if_required(
    output_dir: Annotated[str,Field(description="Directory path to store enzygen input JSON")],
    enzyme_family: Annotated[str, Field(description="Enzyme family of the enzyme to be generated (EC4 category e.g. 1.1.1.1)")],
    pdb: Annotated[str, Field(description="AlphaFold generated PDB file")],
    residues_to_change: Annotated[list[int], Field(description="Indices of residues to be changed (e.g. [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]) - indexing is done from 1")],
    recommended_length: Annotated[int, Field(description="Recommended length of the enzyme to be generated (default: use same length)")] = None,
    add_amino_acids_at_beginning: Annotated[int, Field(description="Number of new amino acids to be added at the beginning (default: 0)")] = 0,
    add_amino_acids_at_end: Annotated[int, Field(description="Number of new amino acids to be added at the end (default: use length to determine the number of amino acids to add)")] = 0,
    add_amino_acids_at_index: Annotated[dict, Field(description="Number of new amino acids to be added at a given index e.g. {\"2\": 10, \"100\": 20} adds 10 amino acids at index 2, 20 at index 100 (keys should be strings representing indices, values should be integers) - indexing is done from 1\n")] = None,
) -> str:
    
    # Handle None or empty dict
    if add_amino_acids_at_index is None:
        add_amino_acids_at_index = {}

    print(enzyme_family)
    print(pdb)
    print(residues_to_change)
    print(add_amino_acids_at_beginning)
    print(add_amino_acids_at_end)
    print(add_amino_acids_at_index)

    with open(pdb, "r") as f:
        content = f.read()
    indices = list(set([int(i) for i in re.findall(r"ATOM\s+\d+\s+\w+\s+\w+\s+\w+\s+(\d+)\s+", content)]))
    if recommended_length is None:
        length = len(indices)
    else:
        length = recommended_length
    motif_indices = []
    motif_seq = []
    motif_coord = []
    cur_index = add_amino_acids_at_beginning
    for i in indices:
        if str(i) in add_amino_acids_at_index.keys():
            cur_index += add_amino_acids_at_index[str(i)]
        if cur_index + i + add_amino_acids_at_end == length + 1:
            break
        if i not in residues_to_change:
            atoms = re.findall(fr"ATOM\s+\d+\s+\w+\s+\w+\s+\w+\s+{i}\s+.*", content)
            atoms = np.array([np.array(a.split()) for a in atoms])
            motif_indices.append(cur_index + i-1)
            motif_seq.append(three_to_one[atoms[0, 3]])
            motif_coord.append(np.round(np.mean(atoms[:, 6:9].astype(float), axis=0), 3).tolist())

    return build_enzygen_input(output_dir,enzyme_family=enzyme_family, motif_indices=motif_indices, motif_seq=motif_seq, motif_coord=motif_coord, recommended_length=length)


@mcp.tool()
def run_enzygen(output_dir: Annotated[str,Field(description="Directory path to store enzygen input JSON")],input_json: Annotated[str, Field(description="Location of script directory")]) -> dict:
    try:
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

        d = f"conv_{datetime.utcnow().strftime('%Y-%m-%d-%H:%M:%S')}"    
        run_enzygen_output = output_dir + f"/run_enygen_{d}"
        os.makedirs(run_enzygen_output)
        print("SUCCESS")
        output_preds = []
        answer_dict = {"status":"successful"}
        for enzyme_family in os.listdir(f"{ENZYGEN_PATH}/outputs"):
            for output in os.listdir(f"{ENZYGEN_PATH}/outputs/{enzyme_family}/pred_pdbs"):
                if output.endswith(".pdb"):
                    with open(f"{ENZYGEN_PATH}/outputs/{enzyme_family}/pred_pdbs/{output}", "r") as f:
                        print("SUCCESS", f"{ENZYGEN_PATH}/outputs/{enzyme_family}/pred_pdbs/{output}")
                        content = f.read()
                        answer_dict["pdb_content"] = content
                    answer_dict["file_path"] = f"{ENZYGEN_PATH}/outputs/{enzyme_family}/pred_pdbs/{output}"
                    # shutil.copy(f"{ENZYGEN_PATH}/outputs/{enzyme_family}/pred_pdbs/{output}", run_enzygen_output)
                    output_preds.append(enzyme_family + "\n" + output + "\n\n" + content)
        with open(f"{ENZYGEN_PATH}/output.log", "r") as f:
            logs = f.read()
        with open(f"{ENZYGEN_PATH}/output.err", "r") as f:
            errors = f.read()
        
      
        answer_dict['answer'] = (
        f"EnzyGen Finished Successfully\n"
        f"Predicted sequence from EnzyGen: {glob.glob(ENZYGEN_PATH + '/outputs/*/protein.txt')[0]}\n"
        f"Predicted structure from EnzyGen: {glob.glob(ENZYGEN_PATH + '/outputs/*/pred_pdbs/*.pdb')[0]}\n\n"
        f"Log File:\n\n----------\n{logs}\n----------\n\n"
        f"Error File:\n\n----------\n{errors}\n----------\n"
    )
        
        answer_dict['visualize'] = "molecule"
        answer_dict["message_render"] = "text"
        return answer_dict
    except Exception as e:
        print("\n========== FULL TRACEBACK ==========")
        traceback.print_exc()
        print("====================================\n")
        # Return error message to MCP
        return {"status": "error", "answer": str(e)}


@mcp.tool()
def run_esp_score_on_enzygen_output(
    sequence_file: Annotated[str, Field(description="Location of enzygen sequence file (protein.txt from enzygen)")],
    ligand_file: Annotated[str, Field(description="Location of ligand file")],
):
    # ESP path
    ESP_DIR = f"{ENZYGEN_PATH}/esp_outputs/{EC_FOLDER}"
    os.makedirs(ESP_DIR, exist_ok=True)
    os.chdir(ESP_DIR)
    
    # if ligand file is in mol
    if ligand_file.endswith(".mol"):
        # convert ligand mol to inchi using obabel, running in docking conda env
        obabel_cmd = f"conda run -n {DOCKING_ENV_NAME} obabel -i mol {ligand_file} -o inchi -O substrate.inchi"
        subprocess.run(obabel_cmd, shell=True, check=True)
        # run ESP prediction in esp conda env
        esp_cmd = f"conda run -n {ESP_CONDA_ENV} python /ocean/projects/cis240137p/eshen3/github/ESP_prediction_function/code/ES_prediction.py {sequence_file} {ESP_DIR}/substrate.inchi"
        result = subprocess.run(esp_cmd, shell=True, check=True, capture_output=True, text=True)
    elif ligand_file.endswith(".inchi") or ligand_file.endswith(".smi"):
        # run ESP prediction in esp conda env
        esp_cmd = f"conda run -n {ESP_CONDA_ENV} python /ocean/projects/cis240137p/eshen3/github/ESP_prediction_function/code/ES_prediction.py {sequence_file} {ligand_file}"
        result = subprocess.run(esp_cmd, shell=True, check=True, capture_output=True, text=True)
    else:
        raise ValueError("Ligand file must be in .mol, .inchi, or .smi format")

    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    
    # Parse only the last line of stdout, which contains the JSON result
    last_line = result.stdout.strip().split('\n')[-1]
    output = json.loads(last_line)
    answer_dict = {'status': 'successful'}
    answer_dict['answer'] = f"ESP Scoring Completed\nESP Score: {output['Prediction']}"
    answer_dict['visualize'] = 'none'
    answer_dict['message_render'] = 'text'
    return answer_dict


@mcp.tool()
def run_colabfold_on_enzygen_output(
    sequence_file: Annotated[str, Field(description="Location of enzygen sequence file (protein.txt from enzygen)")],
    job_time: Annotated[str, Field(description="Time limit for AF2 jobs")] = "1:00:00",
):
    AF2_DIR = f"{ENZYGEN_PATH}/af2_outputs"
    os.makedirs(AF2_DIR, exist_ok=True)
    os.chdir(AF2_DIR)
    
    with open(sequence_file, "r") as f:
        sequence = f.read()
    with open("input_seq.fasta", "w") as f:
        f.write(">enzygen\n" + sequence)

    command = f"apptainer exec --nv -B {COLABFOLD_CACHE}:/cache -B {AF2_DIR}:/work {COLABFOLD_SIF} colabfold_batch /work/input_seq.fasta /work/out --msa-mode mmseqs2_uniref_env --pair-mode unpaired_paired --use-gpu-relax --num-seeds 1 --num-models 1 --model-type alphafold2_ptm"
    with open("submit_colabfold.sh", "w") as f:
        f.write(f"#!/bin/bash\n#SBATCH -N 1\n#SBATCH -p GPU-small\n#SBATCH -t {job_time}\n#SBATCH --gpus=v100-32:1\n#SBATCH --output=output.log\n#SBATCH -n 2\n#SBATCH -e output.err\n{command}")

    p = subprocess.Popen(['sbatch', 'submit_colabfold.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print("Colabfold Job ID:", job_id)
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
    plddt_scores = []
    answer_dict = {'status': 'successful'}
    protein_pdbs = glob.glob(f"{AF2_DIR}/out/*.pdb")
    for pdb in protein_pdbs:
        scores_file = pdb.replace(".pdb", ".json").replace("unrelaxed", "scores")
        with open(scores_file, "r") as f:
            scores = json.loads(f.read())
        plddt_scores.append(float(np.mean(scores["plddt"])))
        answer_dict['file_path'] = pdb
        with open(pdb, "r") as f:
            pdb = f.read()
            answer_dict['pdb_content'] = pdb
        structures.append(f"File: {pdb}\n\n" + pdb)
    
    answer_dict['answer'] = f"Colabfold Job Completed\nAll output files: {protein_pdbs}\nAll plddt scores: {plddt_scores}\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"
    answer_dict['visualize'] = "molecule"
    answer_dict['message_render'] = 'text'
    return answer_dict


@mcp.tool()
def convert_mol_to_sdf_for_docking(
    mol_file: Annotated[str, Field(description="Path to the input MOL file")]
) -> str:
    """
    Converts a MOL file to an SDF file for docking.
    """
    INPUT_DIR = f"{ENZYGEN_PATH}/docking/{EC_FOLDER}"
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.system(f"cp {mol_file} {INPUT_DIR}/ligand.mol")
    os.chdir(INPUT_DIR)

    obabel_cmd = f"conda run -n {DOCKING_ENV_NAME} python /jet/home/eshen3/Agent4Molecule/mcp_agent/util/mol_to_sdf.py --infile {INPUT_DIR}/ligand.mol --out {INPUT_DIR}/ligand_sdf.sdf"
    p = subprocess.Popen(obabel_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    clean_cmd = f"conda run -n {DOCKING_ENV_NAME} python /jet/home/eshen3/Agent4Molecule/mcp_agent/util/clean_fragment.py {INPUT_DIR}/ligand_sdf.sdf {INPUT_DIR}/ligand_cleaned.sdf" 
    p = subprocess.Popen(clean_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    answer_dict = {'status': 'successful'}
    answer_dict["answer"] = f"Ligand sdf file generated at: {INPUT_DIR}/ligand_cleaned.sdf"
    answer_dict["visualize"] = "none"
    answer_dict["message_render"] = "text"
    return answer_dict


@mcp.tool()
def convert_ligand_pdb_to_sdf_for_docking(
    pdb_path: Annotated[str, Field(description="Path to the input PDB file")]
) -> dict:
    """
    Converts a ligand PDB file to an SDF file for docking.
    """
    INPUT_DIR = f"{ENZYGEN_PATH}/docking"
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.system(f"cp {pdb_path} {INPUT_DIR}/ligand.pdb")
    os.chdir(INPUT_DIR)
    
    obabel_cmd = f"obabel {pdb_path} -O {INPUT_DIR}/ligand_sdf.sdf -h"
    p = subprocess.Popen(obabel_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    clean_cmd = f"{PYTHON['vina']} {MOLECULE_AGENT_PATH}/mcp_agent/util/clean_fragment.py {INPUT_DIR}/ligand_sdf.sdf {INPUT_DIR}/ligand_cleaned.sdf" 
    p = subprocess.Popen(clean_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    answer_dict = {"status": "successful"}
    answer_dict["answer"] = f"Ligand sdf file generated at: {INPUT_DIR}/ligand_cleaned.sdf"
    answer_dict["visualize"] = "none"
    answer_dict["message_render"] = "text"
    return answer_dict


def estimate_box_from_ligand(ligand_path: str):
    """
    Estimate center and cubic box size from ligand coordinates.
    """
    mol = next(iter(Chem.SDMolSupplier(ligand_path, removeHs=False)), None)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Cannot read ligand {ligand_path}")

    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    xs, ys, zs = zip(*[(p.x, p.y, p.z) for p in coords])
    cx, cy, cz = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
    size = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) + 10  # Å margin
    return cx, cy, cz, size


@mcp.tool()
def run_docking_pipeline(
    receptor_path: Annotated[str, Field(description="Path to the receptor PDB file")],
    ligand_path: Annotated[str, Field(description="Path to the ligand SDF file")],
    size_x: Annotated[float, Field(description="Size of the search box in the X dimension")] = 80.0,
    size_y: Annotated[float, Field(description="Size of the search box in the Y dimension")] = 80.0,
    size_z: Annotated[float, Field(description="Size of the search box in the Z dimension")] = 80.0,
    center_x: Annotated[float, Field(description="X coordinate of the center of the search box")] = 0.0,
    center_y: Annotated[float, Field(description="Y coordinate of the center of the search box")] = 0.0,
    center_z: Annotated[float, Field(description="Z coordinate of the center of the search box")] = 0.0,
    exhaustiveness: Annotated[int, Field(description="Exhaustiveness of the search (default is 8)")] = 8,
) -> dict:
    """
    Run docking pipeline using AutoDock Vina.
    """
    INPUT_DIR = f"{ENZYGEN_PATH}/docking/{EC_FOLDER}"
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.chdir(INPUT_DIR)

    # -------- Determine docking box automatically if not set --------
    if all(v == 0.0 for v in [center_x, center_y, center_z]) and all(v == 80.0 for v in [size_x, size_y, size_z]):
        cx, cy, cz, size = estimate_box_from_ligand(ligand_path)
        size_x = size_y = size_z = size
        center_x, center_y, center_z = cx, cy, cz
        auto_box = True
        print(f"Auto-determined box center: ({center_x}, {center_y}, {center_z}), size: ({size_x}, {size_y}, {size_z})")
    else:
        auto_box = False

    conda = os.environ.get("CONDA_EXE", "conda")
    prefix = [conda, "run", "-n", "vina"]
    cfg = "receptor_output.box.txt"

    center_x_str = f"{center_x:.6f}"
    center_y_str = f"{center_y:.6f}"
    center_z_str = f"{center_z:.6f}"
    size_x_str = f"{size_x:.6f}"
    size_y_str = f"{size_y:.6f}"
    size_z_str = f"{size_z:.6f}"

    # Prepare receptor
    receptor_cmd = prefix + [
        "mk_prepare_receptor.py",
        "-i", receptor_path,
        "-o", "receptor_output",
        "-p",
        "-v",
        "--box_size", size_x_str, size_y_str, size_z_str,
        "--box_center", center_x_str, center_y_str, center_z_str,
        "-a"
    ]

    p = subprocess.Popen(receptor_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    # Prepare ligand
    ligand_cmd = prefix + [
        "mk_prepare_ligand.py",
        "-i", ligand_path,
        "-o", "ligand_output.pdbqt",
    ]

    p = subprocess.Popen(ligand_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    # Dock molecule
    vina_cmd = prefix + [
        "vina",
        "--receptor", "receptor_output.pdbqt",
        "--ligand", "ligand_output.pdbqt",
        "--config", cfg,
        "--out", "docked.pdbqt",
        "--exhaustiveness", str(exhaustiveness),
    ]

    p = subprocess.Popen(vina_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    with open("docked.pdbqt", "r") as f:
        content = f.read()
        content = re.findall(r"REMARK VINA RESULT:\s+([-+]?\d*\.?\d+)", content)
    binding_affinity = float(content[0])
    # -------- Evaluate quality --------
    quality = (
        "very strong" if binding_affinity <= -10 else
        "strong" if binding_affinity <= -9 else
        "moderate" if binding_affinity <= -6 else
        "weak"
    )

    if binding_affinity > -6.0:
        vina_cmd[-1] = "32"  # increase exhaustiveness
        subprocess.run(vina_cmd, check=True)
        with open("docked.pdbqt", "r") as f:
            energies = re.findall(r"REMARK VINA RESULT:\s+([-+]?\d*\.?\d+)", f.read())
            binding_affinity = float(energies[0])
            quality = (
                "very strong" if binding_affinity <= -10 else
                "strong" if binding_affinity <= -9 else
                "moderate" if binding_affinity <= -7 else
                "weak"
            )

    receptor_pdbqt = os.path.join(INPUT_DIR, "receptor_output.pdbqt")
    docked_pdbqt = os.path.join(INPUT_DIR, "docked.pdbqt")
    auto_determined = " (auto-determined)" if auto_box else ""

    answer_dict = {'status': 'successful'}
    answer_dict["answer"] = (
    f"Successfully finished task. Protein pdbqt file location: {os.path.join(INPUT_DIR, 'receptor_output.pdbqt')}\n"
    f"Docked ligand pdbqt file location: {os.path.join(INPUT_DIR, 'docked.pdbqt')}\n"
    f"Binding Affinity: {binding_affinity}"
)
    answer_dict["visualize"] = "none"
    answer_dict["message_render"] = "text"
    return answer_dict


@mcp.tool()
def get_docked_protein_ligand_complex(
    receptor_pdbqt_path: Annotated[str, Field(description="Receptor pdbqt file")],
    ligand_pdbqt_path: Annotated[str, Field(description="Docked ligand pdbqt file")],
) -> str:
    INPUT_DIR = f"{ENZYGEN_PATH}/docking"
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.system(f"cp {receptor_pdbqt_path} {INPUT_DIR}/receptor.pdbqt")
    os.system(f"cp {ligand_pdbqt_path} {INPUT_DIR}/ligand.pdbqt")
    os.chdir(INPUT_DIR)

    command = f"{PYTHON['diffusion']} {combine_protein_ligand_file} -r {INPUT_DIR}/receptor.pdbqt -l {INPUT_DIR}/ligand.pdbqt -o {INPUT_DIR}/protein_ligand_complex.pdbqt"

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    command = f"obabel {INPUT_DIR}/protein_ligand_complex.pdbqt -O {INPUT_DIR}/protein_ligand_complex.pdb"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    
    answer_dict = {'status': 'successful'}
    answer_dict["answer"] = f"Docked protein-ligand pdb file generated at: {INPUT_DIR}/protein_ligand_complex.pdb"
    answer_dict["visualize"] = "docking"
    answer_dict["pdb_content"] = None
    answer_dict["message_render"] = "text"
    answer_dict["file_path"] = f"{INPUT_DIR}/protein_ligand_complex.pdb"
    
    return answer_dict


@mcp.tool()
def prepare_protein_ligand_complex_for_md(
    protein_path: Annotated[str, Field(description="Path to the protein PDB file generated by ColabFold")],
    ligand_path: Annotated[str, Field(description="Path to the substrate ligand mol file")],
) -> str:
    MD_OUTPUT_DIR = f"{FASTMD_PATH}/md_outputs/{EC_FOLDER}"
    os.makedirs(MD_OUTPUT_DIR, exist_ok=True)
    os.chdir(MD_OUTPUT_DIR)
    os.system(f"cp {protein_path} {MD_OUTPUT_DIR}/protein.pdb")
    os.system(f"cp {ligand_path} {MD_OUTPUT_DIR}/ligand.mol")

    # Convert ligand mol to sdf using obabel
    obabel_cmd = f"conda run -n {DOCKING_ENV_NAME} obabel {MD_OUTPUT_DIR}/ligand.mol -O {MD_OUTPUT_DIR}/ligand.sdf --gen3d"
    p = subprocess.Popen(obabel_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    
    # Convert ligand sdf to pdb 
    sdf2pdb_cmd = f"conda run -n {FASTMD_CONDA_ENV} python {MOLECULE_AGENT_PATH}/mcp_agent/util/gen_ligand.py -i {MD_OUTPUT_DIR}/ligand.sdf -o {MD_OUTPUT_DIR}/ligand.pdb"
    p = subprocess.Popen(sdf2pdb_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    # Fix protein
    fixpdb_cmd = f"conda run -n {FASTMD_CONDA_ENV} python {MOLECULE_AGENT_PATH}/mcp_agent/util/protein_fix.py -i {MD_OUTPUT_DIR}/protein.pdb -o {MD_OUTPUT_DIR}/protein_fixed.pdb"
    p = subprocess.Popen(fixpdb_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    # Merge protein and ligand 
    merge_cmd = f"conda run -n {FASTMD_CONDA_ENV} python {MOLECULE_AGENT_PATH}/mcp_agent/util/merge_complex.py -p {MD_OUTPUT_DIR}/protein_fixed.pdb -l {MD_OUTPUT_DIR}/ligand.pdb -o {MD_OUTPUT_DIR}/complex.pdb"
    p = subprocess.Popen(merge_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    answer_dict = {"status": "successful"}
    answer_dict["answer"] = f"Prepared protein-ligand complex for MD at: {MD_OUTPUT_DIR}/complex.pdb and generated ligand sdf file at {MD_OUTPUT_DIR}/ligand.sdf"
    answer_dict["visualize"] = "none"
    answer_dict["message_render"] = "text"
    
    return answer_dict


@mcp.tool()
def run_fastmd_on_protein_ligand_complex(
    cleaned_complex_path: Annotated[str, Field(description="Path to the cleaned protein-ligand complex PDB file")],
    ligand_path: Annotated[str, Field(description="Path to the substrate ligand sdf file")],
    minimize_steps: Annotated[int, Field(description="Number of minimization steps")] = 500,
    nvt_steps: Annotated[int, Field(description="Number of NVT equilibration steps")] = 50000,
    npt_steps: Annotated[int, Field(description="Number of NPT equilibration steps")] = 50000,
    production_steps: Annotated[int, Field(description="Number of production MD steps")] = 100000,
    temperature_K: Annotated[float, Field(description="Temperature in Kelvin")] = 300,
    job_time: Annotated[str, Field(description="Time limit for MD job")] = "1:00:00",
) -> str:
    """
    Run FastMD simulation on protein-ligand complex.
    """
    MD_OUTPUT_DIR = f"{FASTMD_PATH}/md_outputs/{EC_FOLDER}"
    os.makedirs(MD_OUTPUT_DIR, exist_ok=True)
    os.chdir(MD_OUTPUT_DIR)

    # Build FastMD input YAML file
    yaml_config = f"""project: EnzyGen

defaults:
  engine: openmm
  platform: auto
  temperature_K: {temperature_K}
  timestep_fs: 2.0
  constraints: HBonds
  forcefield: ["amber14-all.xml", "amber14/tip3pfb.xml"]
  ligand_file: {ligand_path}
  ions: NaCl
  box_padding_nm: 1.0
  ionic_strength_molar: 0.15
  neutralize: true

stages:
  - {{ name: minimize, steps: {minimize_steps} }}
  - {{ name: nvt, steps: {nvt_steps}, ensemble: NVT }}
  - {{ name: npt, steps: {npt_steps}, ensemble: NPT }}
  - {{ name: production, steps: {production_steps}, ensemble: NPT }}

systems:
  - id: enzyme_ligand
    fixed_pdb: {cleaned_complex_path}
"""
    
    config_file = os.path.join(MD_OUTPUT_DIR, "fastmd_config.yml")
    with open(config_file, "w") as f:
        f.write(yaml_config)
    print(f"FastMD config written to: {config_file}")

    # Create SLURM submission script
    slurm_script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-small
#SBATCH -t {job_time}
#SBATCH --gpus=v100-32:1
#SBATCH --output=fastmd_output.log
#SBATCH --error=fastmd_output.err
source ~/.bashrc
conda activate {FASTMD_CONDA_ENV}
cd {MD_OUTPUT_DIR}
fastmds simulate -system {config_file}
"""
    
    slurm_file = os.path.join(MD_OUTPUT_DIR, "submit_fastmd.sh")
    with open(slurm_file, "w") as f:
        f.write(slurm_script)
    os.system(f"chmod +x {slurm_file}")

    # Submit job
    p = subprocess.Popen(['sbatch', slurm_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    job_id = extract_job_id(output.decode('utf-8'))
    print(f"FastMD Job ID: {job_id}")
    
    # Monitor job
    while True:
        p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        if job_id not in str(output):
            break
        print("FastMD job still running...")
        time.sleep(60)
    
    # Read output files
    log_file = os.path.join(MD_OUTPUT_DIR, "fastmd_output.log")
    err_file = os.path.join(MD_OUTPUT_DIR, "fastmd_output.err")
    
    logs = ""
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = f.read()
    
    errors = ""
    if os.path.exists(err_file):
        with open(err_file, "r") as f:
            errors = f.read()
    
    answer_dict = {"status": "successful"}
    answer_dict["answer"] = (
        f"FastMD Simulation Completed\n"
        f"Output simulation files are located in the {MD_OUTPUT_DIR}/simulate_output/EnzyGen directory.\n\n"
        f"Log File:\n\n----------\n{logs}\n----------\n\n"
        f"Error File:\n\n----------\n{errors}\n----------\n"
    )
    answer_dict["visualize"] = "none"
    answer_dict["message_render"] = "text"
    return answer_dict


def cleanup():
    os.system(f"rm -rf {ENZYGEN_PATH}/outputs/*")
    os.system(f"rm -rf {ENZYGEN_PATH}/af2_outputs/*")
    os.system(f"rm -rf {ENZYGEN_PATH}/docking/*")
    os.system(f"rm -f {ENZYGEN_PATH}/run_enzygen.sh")
    os.system(f"rm -f {ENZYGEN_PATH}/run_gpu_slurm.sh")
    os.system(f"rm -f {ENZYGEN_PATH}/data/input.json")
    os.system(f"rm -f {ENZYGEN_PATH}/output.log")
    os.system(f"rm -f {ENZYGEN_PATH}/output.err")


if __name__ == "__main__":
    mcp.run(transport='stdio')
