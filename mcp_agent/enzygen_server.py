import os
from typing import Annotated, Dict, List
from pydantic import Field
import json
import subprocess
import time
import pandas as pd
import numpy as np
import glob
import shlex
import random
import re
import logging
import traceback
import sys
from Bio import AlignIO
from util.msa_to_motif import msa_to_enzygen_motif
from rdkit import Chem
from rdkit.Chem import AllChem

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/jet/home/eshen3/Agent4Molecule/mcp_agent/enzygen_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ENZYGEN_PATH = "/ocean/projects/cis240137p/eshen3/github/EnzyGen"
ENZYGEN_CONDA_ENV = "/ocean/projects/cis240137p/eshen3/anaconda3/envs/enzygen/bin/python"
COLABFOLD_CACHE = "/ocean/projects/cis240137p/eshen3/colabfold/cf_cache"
COLABFOLD_SIF = "/ocean/projects/cis240137p/eshen3/colabfold/colabfold_1.5.5-cuda12.2.2.sif"
combine_protein_ligand_file = "/jet/home/eshen3/Agent4Molecule/mcp_agent/util/combine_protein_ligand.py"
PYTHON = {"diffusion": f"/ocean/projects/cis240137p/eshen3/anaconda3/envs/diffusion/bin/python", "vina": f"/ocean/projects/cis240137p/eshen3/anaconda/envs/docking/bin/python"}
DOCKING_ENV_NAME = "docking"
EC_FOLDER = "4.6.1.12"
ESP_CONDA_ENV = "esp"

import sys
sys.path.append(ENZYGEN_PATH)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("enzygen")

three_to_one = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "SEC":"U","PYL":"O","ASX":"B","GLX":"Z","XAA":"X","UNK":"X"
}

def _shquote(s: str) -> str:
    return shlex.quote(s)

def extract_job_id(output: str) -> str:
    """Extracts the job ID from the output of the sbatch command."""
    lines = output.split('\n')
    for line in lines:
        if "Submitted batch job" in line:
            return line.split()[-1]
    return ""

@mcp.tool()
def debug_info() -> str:
    """
    Get debugging information about the current environment and file system.
    """
    try:
        info = []
        info.append(f"Current working directory: {os.getcwd()}")
        info.append(f"Python executable: {sys.executable}")
        info.append(f"Environment PATH: {os.environ.get('PATH', 'Not found')}")
        
        # Check if clustalw is available
        try:
            result = subprocess.run(['which', 'clustalw'], capture_output=True, text=True)
            if result.returncode == 0:
                info.append(f"ClustalW location: {result.stdout.strip()}")
            else:
                info.append("ClustalW: Not found in PATH")
        except Exception as e:
            info.append(f"ClustalW check failed: {e}")
        
        # Check data directory
        data_dir = "/jet/home/eshen3/Agent4Molecule/mcp_agent/data"
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            info.append(f"Data directory contents: {files}")
        else:
            info.append(f"Data directory not found: {data_dir}")
        
        # Check log file
        log_file = "/jet/home/eshen3/Agent4Molecule/mcp_agent/enzygen_server.log"
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            info.append(f"Log file exists: {log_file} ({size} bytes)")
        else:
            info.append(f"Log file not found: {log_file}")
        
        return "\n".join(info)
    
    except Exception as e:
        return f"Debug info failed: {str(e)}\nTraceback: {traceback.format_exc()}"

# AF2 setup
# CONDAPATH = "/ocean/projects/cis240137p/dgarg2/miniconda3"
# HEME_BINDER_PATH = "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/"
# SCRIPT_DIR = os.path.dirname(HEME_BINDER_PATH)
# AF2_script = f"{SCRIPT_DIR}/scripts/af2/af2.py"
# PYTHON = {
#     "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
#     "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
#     "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
#     "general": f"{CONDAPATH}/envs/diffusion/bin/python"
# }
# sys.path.append(HEME_BINDER_PATH)
# sys.path.append(SCRIPT_DIR+"/scripts/utils")
# import utils

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
    start_index: Annotated[int, Field(description="Start index is inclusive (suggestion: only extract 2-5 at a time)")] = 0,
    end_index: Annotated[int, Field(description="End index is exclusive (suggestion: only extract 2-5 at a time)")] = 2,
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
        text = f"- Test {i+1}\n\t- Motif indices: {d['motif']}\n\t- Motif sequence: {np.array(d['seq'])[d['motif']].tolist()}\n\t- Motif coordinates: {np.array(d['coor'])[d['motif']].tolist()}\n\t- Recommended length: {len(d['seq'])}\n\t- Reference PDB: {d['pdb']}"
        # text = f"- Test {i+1}\n\t- Motif indices: {d['motif']}\n\t- Sequence: {d['seq']}\n\t- Coordinates: {d['coor']}\n\t- Recommended length: {len(d['seq'])}\n\t- Reference PDB: {d['pdb']}"
        options.append(text)
    return f"Total motif options: {len(data)}\n\nHere are some mined motifs for the enzyme family {enzyme_category}:\n\n" + "\n".join(options)

# @mcp.tool()
def mine_motifs(
    enzyme_fastafile: Annotated[str, Field(description="Location of MSA fasta file of enzymes in the enzyme category")] = "/jet/home/eshen3/Agent4Molecule/mcp_agent/data/enzyme.fasta",
    ref_pdb_file: Annotated[str, Field(description="Location of representative PDB file of an enzyme in the enzyme category")] = "/jet/home/eshen3/Agent4Molecule/mcp_agent/data/1U3T.pdb",
    chain_id: Annotated[str, Field(description="Chain ID from the PDB file (e.g., 'A', 'B')")] = "A",
    workspace: Annotated[str, Field(description="Location of working directory")] = "/jet/home/eshen3/Agent4Molecule/mcp_agent/data",
):
    """
    Use ClustalW to mine motifs from the MSA of enzymes in a given enzyme category.
    Then call msa_to_enzygen_motif to convert the MSA and a representative PDB chain to EnzyGen motif fields.
    """
    
    logger.info(f"Starting mine_motifs with params: enzyme_file={enzyme_fastafile}, pdb_file={ref_pdb_file}, chain_id={chain_id}, workspace={workspace}")
    
    try:
        # Validate input files exist
        if not os.path.exists(enzyme_fastafile):
            error_msg = f"Enzyme FASTA file not found: {enzyme_fastafile}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
        
        if not os.path.exists(ref_pdb_file):
            error_msg = f"Reference PDB file not found: {ref_pdb_file}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
        
        logger.info(f"Input files validated successfully")
        
        # Setup workspace
        os.makedirs(workspace, exist_ok=True)
        original_cwd = os.getcwd()
        os.chdir(workspace)
        logger.info(f"Changed to workspace directory: {workspace}")

        # Use absolute paths for ClustalW
        abs_enzyme_file = os.path.abspath(enzyme_fastafile)
        aligned_file = os.path.join(workspace, "aligned_enzyme.aln")
        
        logger.info(f"Absolute enzyme file path: {abs_enzyme_file}")
        logger.info(f"Output alignment file: {aligned_file}")

        clustalW_cmd = [
            "clustalw",
            f"-INFILE={abs_enzyme_file}", 
            "-TYPE=PROTEIN",
            f"-OUTFILE={aligned_file}",
        ]
        
        logger.info(f"Running ClustalW command: {' '.join(clustalW_cmd)}")

        try:
            result = subprocess.run(clustalW_cmd, check=True, capture_output=True, text=True)
            logger.info(f"ClustalW stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"ClustalW stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            error_msg = f"ClustalW failed with return code {e.returncode}\nstdout: {e.stdout}\nstderr: {e.stderr}"
            logger.error(error_msg)
            os.chdir(original_cwd)
            return f"ERROR: {error_msg}"
        
        # Check if alignment file was created
        if not os.path.exists(aligned_file):
            error_msg = f"ClustalW alignment file was not created: {aligned_file}"
            logger.error(error_msg)
            os.chdir(original_cwd)
            return f"ERROR: {error_msg}"
        
        logger.info(f"ClustalW alignment completed successfully. File size: {os.path.getsize(aligned_file)} bytes")
        
        # Convert alignment to motif indices, sequences, and coordinates
        logger.info(f"Starting motif extraction with msa_to_enzygen_motif")
        try:
            motif_data = msa_to_enzygen_motif(
                aln_path=aligned_file,
                aln_format="clustal",
                structure_path=ref_pdb_file,
                chain_id=chain_id
            )
            
            logger.info(f"Motif extraction completed successfully")
            logger.debug(f"Motif data keys: {motif_data.keys()}")
            
            motif_indices = motif_data["motif_indices"]
            motif_seq = motif_data["motif_seq"] 
            motif_coord = motif_data["motif_coord"]
            
            logger.info(f"Extracted {len(motif_indices)} motif positions")
            logger.debug(f"Motif indices: {motif_indices}")
            logger.debug(f"Motif sequences: {motif_seq}")

            os.chdir(original_cwd)
            success_msg = f"SUCCESS: Mined motif indices: {motif_indices}\nMined motif sequence: {motif_seq}\nMined motif coordinates: {motif_coord}"
            logger.info("mine_motifs completed successfully")
            return success_msg
        
        except Exception as e:
            error_msg = f"Error in motif processing: {str(e)}\nTraceback: {traceback.format_exc()}"
            logger.error(error_msg)
            os.chdir(original_cwd)
            return f"ERROR: {error_msg}"
    
    except Exception as e:
        error_msg = f"Unexpected error in mine_motifs: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

# @mcp.tool()
# def change_specific_residues_using_enzygen_if_required(
#     enzyme_family: Annotated[str, Field(description="Enzyme family of the enzyme to be generated (EC4 category e.g. 1.1.1.1)")],
#     pdb: Annotated[str, Field(description="AlphaFold generated PDB file")],
#     residues_to_change: Annotated[List[int], Field(description="Indices of residues to be changed (e.g. [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]) - indexing is done from 1")],
#     recommended_length: Annotated[int, Field(description="Recommended length of the enzyme to be generated (default: use same length)")] = None,
#     add_amino_acids_at_beginning: Annotated[int, Field(description="Number of new amino acids to be added at the beginning (default: 0)")] = 0,
#     add_amino_acids_at_end: Annotated[int, Field(description="Number of new amino acids to be added at the end (default: use length to determine the number of amino acids to add)")] = 0,
#     add_amino_acids_at_index: Annotated[str, Field(description="JSON string of amino acids to be added at given indices e.g. '{\"2\": 10, \"100\": 20}' adds 10 amino acids at index 2, 20 at index 100 (indexing from 1). Use empty string '{}' for no additions.")] = "{}",
# ) -> str:
#     with open(pdb, "r") as f:
#         content = f.read()
#     indices = list(set([int(i) for i in re.findall(r"ATOM\s+\d+\s+\w+\s+\w+\s+\w+\s+(\d+)\s+", content)]))
#     if recommended_length is None:
#         length = len(indices)
#     else:
#         length = recommended_length
#     motif_indices = []
#     motif_seq = []
#     motif_coord = []
#     # Parse the JSON string for amino acid additions
#     try:
#         add_amino_acids_dict = json.loads(add_amino_acids_at_index) if add_amino_acids_at_index else {}
#     except json.JSONDecodeError:
#         add_amino_acids_dict = {}
    
#     cur_index = add_amino_acids_at_beginning
#     for i in indices:
#         if str(i) in add_amino_acids_dict.keys():
#             cur_index += add_amino_acids_dict[str(i)]
#         if cur_index + i + add_amino_acids_at_end == length + 1:
#             break
#         if i not in residues_to_change:
#             atoms = re.findall(fr"ATOM\s+\d+\s+\w+\s+\w+\s+\w+\s+{i}\s+.*", content)
#             atoms = np.array([np.array(a.split()) for a in atoms])
#             motif_indices.append(cur_index + i-1)
#             motif_seq.append(three_to_one[atoms[0, 3]])
#             motif_coord.append(np.round(np.mean(atoms[:, 6:9].astype(float), axis=0), 3).tolist())
#     # return f"Data for enzygen:\n- Motif indices: {motif_indices}\n- Motif sequence: {motif_seq}\n- Motif coordinates: {motif_coord}\n- Recommended length: {length}"
#     return build_enzygen_input(enzyme_family=enzyme_family, motif_indices=motif_indices, motif_seq=motif_seq, motif_coord=motif_coord, recommended_length=length)


@mcp.tool()
def build_enzygen_input(
    enzyme_family: Annotated[str, Field(description="Enzyme family of the enzyme to be generated (EC4 category e.g. 1.1.1.1)")],
    motif_indices: Annotated[list[int], Field(description="Indices of the motif")],
    motif_seq: Annotated[list[str], Field(description="Sequence of the motif")],
    motif_coord: Annotated[list[list[float]], Field(description="Coordinates of the motif")],
    recommended_length: Annotated[int, Field(description="Recommended length of the enzyme to be generated")],
    ref_pdb_chain: Annotated[str, Field(description="Reference PDB chain")] = "AAAA.A",
) -> str:
    os.makedirs(ENZYGEN_PATH+"data", exist_ok=True)
    file_name = os.path.join(ENZYGEN_PATH, "data/input.json")
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
            # seq += random.choice(amino_acids)
            coord += "0.0,0.0,0.0,"
    if coord[-1] == ",":
        coord = coord[:-1]
    # seq = "".join(motif_seq)
    # coord = ",".join([",".join([str(j) for j in i]) for i in motif_coord])

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
    return f"ESP Scoring Completed\nESP Score: {output['Prediction']}"


@mcp.tool()
def run_colabfold_on_enzygen_output(
    sequence_file: Annotated[str, Field(description="Location of enzygen sequence file (protein.txt from enzygen)")],
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
    protein_pdbs = glob.glob(f"{AF2_DIR}/out/*.pdb")
    for pdb in protein_pdbs:
        scores_file = pdb.replace(".pdb", ".json").replace("unrelaxed", "scores")
        with open(scores_file, "r") as f:
            scores = json.loads(f.read())
        plddt_scores.append(float(np.mean(scores["plddt"])))
        with open(pdb, "r") as f:
            pdb = f.read()
        structures.append(f"File: {pdb}\n\n" + pdb)
    
    return f"Colabfold Job Completed\nAll output files: {protein_pdbs}\nAll plddt scores: {plddt_scores}\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"

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

    return f"Ligand sdf file generated at: {INPUT_DIR}/ligand_cleaned.sdf"

@mcp.tool()
def convert_ligand_pdb_to_sdf_for_docking(
    pdb_path: Annotated[str, Field(description="Path to the input PDB file")]
) -> str:
    INPUT_DIR = f"{ENZYGEN_PATH}/docking/{EC_FOLDER}"
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.system(f"cp {pdb_path} {INPUT_DIR}/ligand.pdb")
    os.chdir(INPUT_DIR)

    obabel_cmd = f"conda run -n {DOCKING_ENV_NAME} obabel {pdb_path} -O {INPUT_DIR}/ligand_sdf.sdf -h"
    p = subprocess.Popen(obabel_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    clean_cmd = f"conda run -n {DOCKING_ENV_NAME} python /jet/home/eshen3/Agent4Molecule/mcp_agent/util/clean_fragment.py {INPUT_DIR}/ligand_sdf.sdf {INPUT_DIR}/ligand_cleaned.sdf" 
    p = subprocess.Popen(clean_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    return f"Ligand sdf file generated at: {INPUT_DIR}/ligand_cleaned.sdf"

def estimate_box_from_ligand(ligand_path: str):
    """Estimate center and cubic box size from ligand coordinates."""
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
) -> str:
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
    prefix = [conda, "run", "-n", "docking"]
    cfg = "receptor_output.box.txt"

    # Format coordinates to avoid scientific notation (which can confuse argument parsers when negative)
    # Use format with sufficient precision but without scientific notation
    center_x_str = f"{center_x:.6f}"
    center_y_str = f"{center_y:.6f}"
    center_z_str = f"{center_z:.6f}"
    size_x_str = f"{size_x:.6f}"
    size_y_str = f"{size_y:.6f}"
    size_z_str = f"{size_z:.6f}"

    # prepare receptor
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
    # Use subprocess.run instead of Popen for better handling
    result = subprocess.run(receptor_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mk_prepare_receptor.py failed with return code {result.returncode}: {result.stderr}")

    # prepare ligand
    ligand_cmd = prefix + [
        "mk_prepare_ligand.py",
        "-i", ligand_path,
        "-o", "ligand_output.pdbqt",
    ]
    print(f"Running: {' '.join(shlex.quote(arg) for arg in ligand_cmd)}")
    result = subprocess.run(ligand_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"mk_prepare_ligand.py failed with return code {result.returncode}: {result.stderr}")
    # print(output, err)

    # dock molecule
    vina_cmd = prefix + [
        "vina",
        "--receptor", "receptor_output.pdbqt",
        "--ligand", "ligand_output.pdbqt",
        "--config", cfg,
        "--out", "docked.pdbqt",
        "--exhaustiveness", str(exhaustiveness),
    ]
    # print(" ".join(vina_cmd))
    p = subprocess.Popen(vina_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    # print(output, err)
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

    # return f"Successfully finished task. Protein pdbqt file location: {os.path.join(INPUT_DIR, "receptor_output.pdbqt")}\nDocked ligand pdbqt file location: {os.path.join(INPUT_DIR, "docked.pdbqt")}\nBinding Affinity: {binding_affinity}\n\nLogs:\n----\n{output.decode('utf-8')}\n----\nErrors:\n----\n{err.decode('utf-8')}\n----"
    return f"Successfully finished task. Protein pdbqt file location: {os.path.join(INPUT_DIR, "receptor_output.pdbqt")}\nDocked ligand pdbqt file location: {os.path.join(INPUT_DIR, "docked.pdbqt")}\nBinding Affinity: {binding_affinity}\nDocking score is {quality}\nDocking box is {size_x} x {size_y} x {size_z} centered at ({center_x}, {center_y}, {center_z}){' (auto-determined)' if auto_box else ''}."


@mcp.tool()
def get_docked_protein_ligand_complex(
    receptor_pdbqt_path: Annotated[str, Field(description="Receptor pdbqt file")],
    ligand_pdbqt_path: Annotated[str, Field(description="Docked ligand pdbqt file")],
) -> str:
    INPUT_DIR = f"{ENZYGEN_PATH}/docking/{EC_FOLDER}"
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.system(f"cp {receptor_pdbqt_path} {INPUT_DIR}/receptor.pdbqt")
    os.system(f"cp {ligand_pdbqt_path} {INPUT_DIR}/ligand.pdbqt")
    os.chdir(INPUT_DIR)

    command = f"{PYTHON["diffusion"]} {combine_protein_ligand_file} -r {INPUT_DIR}/receptor.pdbqt -l {INPUT_DIR}/ligand.pdbqt -o {INPUT_DIR}/protein_ligand_complex.pdbqt"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    command = f"conda run -n {DOCKING_ENV_NAME} obabel {INPUT_DIR}/protein_ligand_complex.pdbqt -O {INPUT_DIR}/protein_ligand_complex.pdb"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
        
    return f"Docked protein-ligand pdb file generated at: {INPUT_DIR}/protein_ligand_complex.pdb"

# def combine_pdbs(protein_path, ligand_path, output_path="complex.pdb"):
#     """
#     Combine a receptor (protein) PDB and a ligand PDB into one complex file.

#     Args:
#         protein_path (str): Path to receptor PDB (ATOM records, e.g., chain A).
#         ligand_path (str): Path to ligand PDB (HETATM or ATOM records labeled LIG).
#         output_path (str): Path to save combined PDB (default: 'complex.pdb').
#     """
#     with open(protein_path, "r") as f:
#         protein_lines = [l for l in f.readlines() if l.startswith(("ATOM", "TER"))]

#     with open(ligand_path, "r") as f:
#         ligand_lines = [l for l in f.readlines() if l.startswith(("HETATM", "ATOM", "CONECT"))]

#     # Avoid duplicate END or TER lines
#     protein_lines = [l for l in protein_lines if not l.startswith("END")]
#     ligand_lines = [l for l in ligand_lines if not l.startswith("END")]

#     with open(output_path, "w") as out:
#         out.writelines(protein_lines)
#         out.write("TER\n")  # mark end of protein chain
#         out.writelines(ligand_lines)
#         out.write("END\n")

# @mcp.tool()
# def run_gromacs_copilot(
#     prompt: Annotated[str, Field(description="Natural language prompt to control GROMACS Copilot")],
#     substrate_ligand_pdb_path: Annotated[str, Field(description="Path to the protein-ligand complex PDB file")],
#     receptor_pdb_path: Annotated[str, Field(description="Path to the receptor PDB file")],
#     api_key: Annotated[str, Field(description="API key for LLM service")] = os.getenv("GEMINI_API_KEY"),
#     model: Annotated[str, Field(description="LLM model name, e.g., gpt-4o, deepseek-chat, gemini-2.0-flash")] = "gemini-2.0-flash",
#     api_url: Annotated[str, Field(description="URL for LLM API")] = "https://generativelanguage.googleapis.com/v1beta/chat/completions",
#     mode: Annotated[str, Field(description="Copilot mode: copilot, agent, or debug")] = "agent",
#     ) -> str:
#     """
#     Submits a SLURM job to run GROMACS Copilot and waits for completion.
#     """

#     INPUT_DIR = f"{ENZYGEN_PATH}/docking"
#     os.makedirs(INPUT_DIR, exist_ok=True)
#     os.system(f"cp {substrate_ligand_pdb_path} {INPUT_DIR}/substrate_ligand.pdb")
#     os.system(f"cp {receptor_pdb_path} {INPUT_DIR}/receptor.pdb")
#     os.chdir(INPUT_DIR)
#     # combine_pdbs(f"{INPUT_DIR}/receptor.pdb", f"{INPUT_DIR}/substrate_ligand.pdb", f"{INPUT_DIR}/protein.pdb")

#     slurm_script = os.path.join(INPUT_DIR, "run_copilot.sh")
#     log_file = os.path.join(INPUT_DIR, "copilot_output.log")

#     # Build the gmx_copilot command (quote everything)
#     cmd = (
#         f"gmx_copilot "
#         f"--workspace {_shquote(INPUT_DIR)} "
#         f"--prompt {_shquote(prompt)} "
#         f"--api-key {api_key} "
#         f"--model {_shquote(model)} "
#         f"--url {_shquote(api_url)} "
#         f"--mode {_shquote(mode)}"
#     )

#     script_text = (
#         "#!/bin/bash\n"
#         f"#SBATCH -N 1\n"
#         f"#SBATCH -p GPU-shared\n"
#         f"#SBATCH -t 24:00:00\n"
#         f"#SBATCH --gres=gpu:1\n"
#         f"#SBATCH --output={log_file}\n\n"
#         "source ~/.bashrc\n"
#         'eval "$(conda shell.bash hook)"\n'
#         f'echo "=== Activating conda env"\n'
#         f"conda activate gromacs_env\n\n"
#         "nvidia-smi\n"
#         'echo "=== Setting project path"\n'
#         f"cd /jet/home/eshen3/gromacs_copilot\n"
#         'echo "=== Running gmx_copilot"\n\n'
#         f"{cmd}\n"
#     )

#     with open(slurm_script, "w") as f:
#         f.write(script_text)
#     print(f"SLURM script written to {slurm_script}")

#     p = subprocess.Popen(["sbatch", slurm_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     output, err = p.communicate()

#     if p.returncode != 0:
#         raise RuntimeError(f"Failed to submit SLURM job:\n{err.decode()}")

#     output_str = output.decode("utf-8")
#     print(output_str)
#     job_id = output_str.strip().split()[-1]

#     # Wait for job to complete
#     print(f"Submitted job {job_id}. Waiting for it to complete...")
#     while True:
#         q = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         qout, _ = q.communicate()
#         if job_id not in qout.decode("utf-8"):
#             break
#         print("Job still running...")
#         time.sleep(60)


#     # Collect and return outputs
#     with open(log_file, "r") as f:
#         logs = f.read()

#     return f"Job {job_id} completed.\n\nLog Output:\n{logs}\n\nErrors:\n{errors}"


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

    # run_esp_score_on_enzygen_output(
    #     "/ocean/projects/cis240137p/eshen3/github/EnzyGen/outputs/1.1.1/protein.txt",
    #     "/jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/1.1.1.270/substrate_ligand.mol"
    # )

    # convert_mol_to_sdf_for_docking(
    #     mol_file="/jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/1.2.1.10/substrate_ligand.mol"
    # )

    # result = run_docking_pipeline(
    #     receptor_path="/ocean/projects/cis240137p/eshen3/github/EnzyGen/af2_outputs//out/enzygen_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb",
    #     ligand_path="/ocean/projects/cis240137p/eshen3/github/EnzyGen/docking/1.1.1.201/ligand_cleaned.sdf"
    # )
    # print(result)

    # get_docked_protein_ligand_complex(
    #     receptor_pdbqt_path="/ocean/projects/cis240137p/eshen3/github/EnzyGen/docking/receptor_output.pdbqt",
    #     ligand_pdbqt_path="/ocean/projects/cis240137p/eshen3/github/EnzyGen/docking/docked.pdbqt")

    # run_gromacs_copilot(
    #     prompt = "Run a 10ns molecular dynamics simulation on ligand substrate_ligand.pdb and receptor in receptor.pdb. Assume ligand name is UNL.",
    #     substrate_ligand_pdb_path = "/jet/home/eshen3/Agent4Molecule/mcp_agent/inputs/substrate_ligand.pdb",
    #     receptor_pdb_path = "/ocean/projects/cis240137p/eshen3/github/EnzyGen/docking/receptor.pdb",
    #     api_key = os.getenv("GEMINI_API_KEY")
    # )

    # run_enzygen("/ocean/projects/cis240137p/eshen3/github/EnzyGen/data/input.json")
    
    # print(find_enzyme_category_using_keywords(["oxidase", "D-ARABINONO-1,4-LACTONE"]))
    # print(find_enzyme_category_using_keywords(["adenylate", "cyclase", "adenylylcyclase"]))
    # print(find_mined_motifs_enzyme_category("4.6.1.1", start_index=0, end_index=1))
    # cleanup()
    # print(build_enzygen_input("4.6.1.1", [7, 9, 13, 16, 19, 20, 22, 28, 58, 59, 63, 67, 75, 82, 84, 90, 115, 117, 119, 123, 130, 138, 143, 149, 154, 158, 167], ['R', 'I', 'F', 'I', 'F', 'T', 'M', 'S', 'G', 'D', 'A', 'A', 'E', 'A', 'A', 'A', 'R', 'G', 'H', 'A', 'S', 'A', 'V', 'L', 'A', 'I', 'Y'], [[43.506, -6.758, 46.718], [38.345, -4.765, 44.212], [27.09, 0.648, 48.385], [17.123, 3.522, 50.054], [11.402, 2.391, 54.192], [10.907, 3.571, 57.767], [6.486, 3.519, 54.572], [5.508, -5.321, 61.885], [19.634, -1.77, 57.845], [18.156, -1.1, 54.433], [28.436, -5.703, 47.564], [36.463, -11.125, 46.134], [34.608, -7.277, 34.653], [24.991, -3.646, 35.749], [23.48, -0.824, 40.253], [13.852, -3.122, 40.484], [21.286, 5.9, 46.057], [28.037, 4.6, 45.147], [34.503, 2.711, 43.701], [41.431, -2.971, 49.986], [38.431, -12.864, 66.38], [36.711, -5.303, 52.874], [35.782, 1.19, 51.082], [30.113, 8.463, 47.15], [21.426, 12.621, 44.494], [24.48, 6.126, 40.615], [35.757, 1.433, 32.015]], 198, "1wc3.A"))
    # print(build_enzygen_input("4.6.1.1", [0, 1, 4], ['D', 'I', 'G'], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], 5))
    # print(run_enzygen(f"{ENZYGEN_PATH}/data/input.json"))
    # print(run_enzygen(f"{ENZYGEN_PATH}/data/test_2.json"))
    # print(run_af2_on_enzygen_output("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/outputs/4.6.1/protein.txt"))

    # print(run_enzygen(f"{ENZYGEN_PATH}/data/new_test.json"))
    # print(run_af2_on_enzygen_output("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/outputs/1.1.1/protein.txt"))

    # print(find_mined_motifs_enzyme_category("3.1.1.2", start_index=0, end_index=1))
    # print(build_enzygen_input(
    #     "3.1.1.2",
    #     [24, 103, 105, 110, 114, 123, 132, 133, 139, 152, 153, 154, 156, 163, 164, 169, 193, 212, 228, 238, 242, 247, 250, 255, 263, 265, 267, 275, 285, 297],
    #     ['M', 'P', 'L', 'D', 'P', 'E', 'V', 'R', 'N', 'F', 'L', 'Q', 'V', 'Y', 'Y', 'K', 'A', 'N', 'I', 'I', 'D', 'F', 'T', 'K', 'Y', 'Q', 'F', 'Q', 'E', 'I', 'R', 'Q', 'K', 'V', 'N', 'E', 'L', 'L', 'A', 'K', 'A', 'V', 'P', 'K', 'D', 'P', 'V', 'G', 'E', 'T', 'R', 'D', 'M', 'K', 'I', 'K', 'L', 'E', 'D', 'Y', 'E', 'L', 'P', 'I', 'R', 'I', 'Y', 'S', 'P', 'I', 'K', 'R', 'T', 'N', 'N', 'G', 'L', 'V', 'M', 'H', 'F', 'H', 'G', 'G', 'A', 'W', 'I', 'L', 'G', 'S', 'I', 'E', 'T', 'E', 'D', 'A', 'I', 'S', 'R', 'I', 'L', 'S', 'N', 'S', 'C', 'E', 'C', 'T', 'V', 'I', 'S', 'V', 'D', 'Y', 'R', 'L', 'A', 'P', 'E', 'Y', 'K', 'F', 'P', 'T', 'A', 'V', 'Y', 'D', 'C', 'F', 'N', 'A', 'I', 'V', 'W', 'A', 'R', 'D', 'N', 'A', 'G', 'E', 'L', 'G', 'I', 'D', 'K', 'D', 'K', 'I', 'A', 'T', 'F', 'G', 'I', 'S', 'A', 'G', 'G', 'N', 'L', 'V', 'A', 'A', 'T', 'S', 'L', 'L', 'A', 'R', 'D', 'N', 'K', 'L', 'K', 'L', 'T', 'A', 'Q', 'V', 'P', 'V', 'V', 'P', 'F', 'V', 'Y', 'L', 'D', 'L', 'A', 'S', 'K', 'S', 'M', 'N', 'R', 'Y', 'R', 'K', 'G', 'Y', 'F', 'L', 'D', 'I', 'N', 'L', 'P', 'V', 'D', 'Y', 'G', 'V', 'K', 'M', 'Y', 'I', 'R', 'D', 'E', 'K', 'D', 'L', 'Y', 'N', 'P', 'L', 'F', 'S', 'P', 'L', 'I', 'A', 'E', 'D', 'L', 'S', 'N', 'L', 'P', 'Q', 'A', 'I', 'V', 'V', 'T', 'A', 'E', 'Y', 'D', 'P', 'L', 'R', 'D', 'Q', 'G', 'E', 'A', 'Y', 'A', 'Y', 'R', 'L', 'M', 'E', 'S', 'G', 'V', 'P', 'T', 'L', 'S', 'F', 'R', 'V', 'N', 'G', 'N', 'V', 'H', 'A', 'F', 'L', 'G', 'S', 'P', 'R', 'T', 'S', 'R', 'Q', 'V', 'T', 'V', 'M', 'I', 'G', 'A', 'L', 'L', 'K', 'D', 'I', 'F', 'K'],
    #     [[47.353, 3.866, 89.348], [44.679, 1.176, 89.838], [42.494, 1.73, 86.774], [40.544, -0.677, 84.653], [43.146, -2.228, 82.241], [41.084, -1.249, 79.148], [41.034, 2.345, 80.413], [44.765, 2.304, 81.143], [45.455, 1.013, 77.61], [43.226, 3.766, 76.182], [45.247, 6.452, 78.02], [48.531, 5.116, 76.603], [47.165, 5.3, 73.053], [45.499, 8.684, 73.628], [48.629, 10.528, 74.775], [50.822, 8.826, 72.111], [48.359, 10.03, 69.444], [48.974, 13.69, 70.327], [45.603, 15.066, 69.174], [45.625, 18.328, 71.179], [48.494, 20.599, 70.226], [48.304, 24.105, 68.885], [51.828, 23.825, 67.348], [50.518, 21.69, 64.467], [46.743, 22.076, 64.343], [44.339, 25.04, 64.427], [41.481, 25.106, 66.932], [38.805, 24.025, 64.459], [40.79, 20.926, 63.461], [41.301, 20.023, 67.159], [37.551, 20.422, 67.731], [36.836, 18.073, 64.825], [39.539, 15.548, 65.755], [38.217, 15.235, 69.327], [34.602, 15.056, 68.076], [35.549, 12.071, 65.845], [37.159, 10.401, 68.933], [34.055, 10.895, 71.082], [31.614, 9.849, 68.36], [33.374, 6.472, 67.837], [33.662, 5.728, 71.596], [30.441, 3.747, 72.201], [28.656, 1.561, 69.599], [25.221, 2.706, 68.34], [22.177, 0.661, 69.307], [19.136, 0.74, 67.044], [16.095, 2.796, 68.111], [12.499, 3.174, 66.906], [13.006, 6.424, 65.006], [15.282, 9.485, 64.65], [14.178, 12.918, 63.481], [16.368, 16.004, 63.031], [14.827, 19.474, 63.18], [15.429, 23.068, 64.281], [13.786, 25.153, 67.004], [13.292, 28.746, 65.866], [14.821, 31.257, 68.317], [14.765, 35.028, 67.63], [18.508, 35.152, 66.727], [19.223, 31.484, 65.859], [17.884, 28.112, 64.67], [18.699, 25.569, 67.382], [19.202, 22.063, 65.969], [17.76, 19.091, 67.857], [17.415, 15.352, 67.33], [14.355, 13.478, 68.622], [14.883, 9.793, 69.509], [11.982, 7.376, 69.695], [12.714, 4.125, 71.546], [12.418, 0.598, 70.151], [9.629, 0.031, 72.69], [7.805, 3.041, 74.135], [6.106, 2.257, 77.471], [5.744, 5.704, 79.09], [5.05, 9.358, 78.309], [8.233, 10.737, 79.848], [10.803, 12.954, 78.197], [14.549, 13.262, 78.689], [16.397, 16.369, 77.52], [19.976, 15.468, 76.703], [22.753, 18.058, 76.699], [26.133, 16.805, 75.397], [29.498, 17.529, 77.019], [32.761, 18.82, 75.534], [33.898, 21.627, 77.848], [31.528, 24.194, 76.32], [33.846, 24.339, 73.235], [33.265, 21.066, 71.366], [30.957, 18.053, 71.053], [27.518, 17.792, 69.506], [24.444, 15.62, 69.053], [26.62, 13.278, 66.948], [29.298, 12.872, 69.626], [26.819, 11.603, 72.212], [24.37, 10.005, 69.727], [25.203, 6.497, 71.022], [24.394, 7.599, 74.587], [21.038, 9.047, 73.531], [20.074, 5.783, 71.821], [20.926, 3.577, 74.811], [19.235, 6.081, 77.141], [16.045, 6.172, 75.105], [15.797, 2.369, 74.959], [16.515, 1.953, 78.681], [14.043, 4.684, 79.66], [11.588, 3.399, 77.036], [10.741, 7.0, 76.336], [11.443, 9.898, 74.026], [14.844, 11.608, 74.192], [15.714, 15.014, 72.709], [19.372, 15.894, 72.206], [20.258, 19.603, 71.863], [23.012, 21.278, 69.763], [23.479, 24.334, 71.99], [25.85, 27.061, 70.847], [29.484, 26.69, 71.84], [32.4, 28.886, 72.946], [34.585, 30.707, 71.966], [32.029, 31.87, 69.346], [29.22, 32.159, 71.908], [30.259, 32.791, 75.466], [28.475, 32.172, 78.736], [25.535, 32.316, 79.315], [24.47, 31.431, 75.711], [24.715, 27.621, 76.135], [22.662, 27.753, 79.349], [19.849, 29.712, 77.733], [19.915, 27.598, 74.554], [19.391, 24.48, 76.648], [16.69, 26.119, 78.772], [14.771, 27.396, 75.703], [14.877, 23.847, 74.292], [13.161, 22.59, 77.535], [10.529, 25.324, 77.267], [9.994, 24.365, 73.567], [9.515, 20.687, 74.495], [6.985, 21.595, 77.2], [5.126, 23.894, 74.775], [5.016, 21.031, 72.206], [4.228, 18.285, 74.775], [0.992, 17.202, 73.087], [2.7, 17.023, 69.67], [5.481, 14.879, 71.165], [3.04, 12.647, 73.088], [4.637, 13.443, 76.44], [3.425, 14.46, 79.89], [4.39, 18.014, 80.864], [5.12, 16.901, 84.464], [7.449, 14.07, 83.301], [10.315, 16.088, 81.798], [13.838, 15.159, 82.976], [17.279, 16.518, 82.116], [20.491, 14.551, 81.465], [24.03, 15.642, 80.691], [27.704, 14.63, 80.811], [30.586, 16.882, 81.938], [29.8, 20.393, 80.548], [26.417, 18.792, 79.852], [26.191, 18.058, 83.571], [27.002, 21.718, 84.242], [24.091, 22.676, 81.942], [21.761, 20.278, 83.786], [22.561, 21.971, 87.138], [22.363, 25.49, 85.637], [19.055, 24.711, 83.911], [17.529, 23.473, 87.141], [18.182, 26.871, 88.726], [16.534, 28.673, 85.806], [13.687, 26.167, 85.95], [13.127, 27.05, 89.591], [13.329, 30.83, 88.944], [10.713, 30.54, 86.205], [8.562, 27.998, 88.074], [9.051, 25.408, 85.324], [8.692, 21.922, 86.801], [11.186, 19.196, 85.983], [10.644, 15.745, 87.363], [14.342, 14.851, 87.545], [17.93, 15.905, 86.797], [20.842, 13.654, 85.931], [24.226, 15.397, 85.932], [26.894, 12.861, 85.091], [30.459, 13.833, 85.967], [29.411, 17.491, 86.086], [31.381, 20.715, 86.366], [29.721, 22.78, 89.138], [32.491, 24.975, 90.576], [36.02, 26.352, 90.164], [37.504, 23.755, 92.476], [41.15, 24.38, 93.363], [41.288, 22.677, 96.758], [39.97, 19.111, 96.435], [42.069, 15.973, 96.368], [40.533, 14.669, 93.133], [41.36, 17.941, 91.344], [44.986, 18.115, 92.426], [45.732, 14.42, 91.94], [43.972, 13.653, 88.673], [44.255, 16.962, 86.824], [46.95, 15.476, 84.571], [47.361, 12.048, 82.945], [43.731, 10.944, 83.009], [42.344, 12.34, 79.74], [40.661, 15.41, 81.257], [43.734, 17.602, 81.522], [43.106, 20.768, 83.58], [45.484, 23.68, 84.316], [45.16, 25.785, 87.484], [43.928, 28.252, 88.578], [41.733, 28.827, 85.496], [41.86, 26.196, 82.723], [41.601, 26.737, 78.981], [37.993, 25.623, 78.636], [36.675, 27.993, 81.308], [38.63, 30.994, 79.902], [37.045, 30.372, 76.479], [33.516, 30.009, 77.894], [33.402, 32.857, 80.452], [33.43, 36.577, 79.597], [35.52, 37.706, 82.557], [36.431, 36.785, 86.142], [33.076, 38.049, 87.571], [31.331, 35.056, 85.914], [33.329, 32.672, 88.133], [30.905, 33.297, 91.04], [27.805, 33.366, 88.871], [25.529, 30.515, 89.983], [24.543, 29.71, 86.349], [28.226, 29.024, 85.719], [29.033, 27.592, 89.185], [25.83, 26.22, 90.645], [27.685, 25.004, 93.75], [28.071, 28.663, 94.84], [24.318, 29.163, 94.801], [22.879, 30.438, 98.106], [19.845, 28.171, 97.795], [19.795, 24.641, 96.296], [16.356, 23.56, 97.644], [13.187, 22.995, 95.567], [15.242, 21.567, 92.774], [14.345, 18.376, 90.96], [15.092, 14.941, 92.333], [18.725, 14.217, 91.351], [20.89, 11.349, 90.177], [24.545, 12.534, 90.401], [27.197, 10.211, 89.016], [30.924, 10.638, 89.515], [34.099, 8.779, 88.567], [36.946, 7.899, 90.967], [39.924, 9.136, 88.914], [38.211, 12.241, 87.594], [39.422, 15.656, 88.697], [35.794, 16.833, 88.927], [34.854, 13.941, 91.257], [34.835, 15.875, 94.538], [32.831, 18.911, 93.304], [30.164, 16.585, 91.877], [29.889, 14.673, 95.18], [29.74, 18.05, 96.963], [26.832, 19.182, 94.759], [24.963, 15.981, 95.522], [25.477, 16.451, 99.274], [24.335, 20.069, 99.046], [21.107, 18.911, 97.392], [20.563, 16.271, 100.125], [21.115, 18.861, 102.801], [18.624, 21.19, 101.047], [15.91, 18.545, 101.128], [16.05, 17.286, 97.575], [15.816, 13.474, 97.231], [19.148, 12.456, 95.656], [21.076, 9.388, 94.649], [24.79, 9.891, 94.257], [27.219, 7.176, 93.347], [30.848, 7.042, 92.34], [32.041, 4.673, 89.571], [35.379, 3.22, 90.639], [38.411, 2.471, 88.418], [37.251, 4.941, 85.81], [38.301, 8.281, 84.511], [36.172, 11.232, 83.482], [33.115, 10.258, 81.4], [34.124, 6.553, 81.516], [36.45, 7.008, 78.549], [38.343, 3.931, 77.418], [35.554, 1.71, 78.763], [32.511, 1.464, 76.509], [31.192, -1.391, 78.713], [31.087, 0.905, 81.791], [29.643, 3.767, 79.743], [26.819, 1.53, 78.63], [26.131, 0.122, 82.118], [25.892, 3.575, 83.674], [23.426, 4.731, 80.963], [21.306, 1.593, 81.455], [21.256, 2.056, 85.278], [20.185, 5.667, 84.889], [17.474, 4.651, 82.394], [16.054, 2.081, 84.801], [15.761, 4.769, 87.513], [14.197, 7.338, 85.174], [11.782, 4.719, 83.86], [10.848, 4.082, 87.485], [10.249, 7.776, 88.122], [8.138, 7.979, 84.912], [5.823, 5.148, 86.13]],
    #     306,
    #     "5l2p.A"
    # ))
    # print(run_enzygen(f"{ENZYGEN_PATH}/data/input.json"))
    # print(run_af2_on_enzygen_output("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/outputs/3.1.1/protein.txt"))

    # print(run_colabfold_on_enzygen_output("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/outputs/1.1.1/protein.txt"))

    # print(run_docking_pipeline(
    #     size_z=40,
    #     center_x=0,
    #     size_x=40,
    #     center_y=0,
    #     size_y=40,
    #     center_z=0,
    #     ligand_path="/ocean/projects/cis240137p/dgarg2/github/EnzyGen//docking/ligand_cleaned.sdf",
    #     receptor_path="/ocean/projects/cis240137p/dgarg2/github/EnzyGen/af2_outputs/out/enzygen_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb",
    # ))
    # print(get_docked_protein_ligand_complex(
    #     receptor_pdbqt_path="/ocean/projects/cis240137p/dgarg2/github/EnzyGen//docking/receptor_output.pdbqt",
    #     ligand_pdbqt_path="/ocean/projects/cis240137p/dgarg2/github/EnzyGen//docking/docked.pdbqt",
    # ))

    # print(change_specific_residues_using_enzygen_if_required(
    #     enzyme_family="4.6.1.1", 
    #     pdb="/ocean/projects/cis240137p/dgarg2/github/EnzyGen/af2_outputs/out/enzygen_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb", 
    #     residues_to_change=[5, 6, 7], 
    #     recommended_length=100, 
    #     add_amino_acids_at_beginning=2, 
    #     add_amino_acids_at_end=0,
    #     add_amino_acids_at_index={"1": 50, "3": 20}
    # ))
