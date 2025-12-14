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
EC_FOLDER = "2.4.1.135"
ESP_CONDA_ENV = "esp"
FASTMD_PATH = "/ocean/projects/cis240137p/eshen3/github/FastMDSimulation"
AGENT4MOLECULE_PATH = "/jet/home/eshen3/Agent4Molecule/"
FASTMD_CONDA_ENV = "fastmds"

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
    
    # Get output files
    sequence_files = glob.glob(f"{ENZYGEN_PATH}/outputs/*/protein.txt")
    structure_files = glob.glob(f"{ENZYGEN_PATH}/outputs/*/pred_pdbs/*.pdb")
    
    sequence_file = sequence_files[0] if sequence_files else "Not found"
    structure_file = structure_files[0] if structure_files else "Not found"
    
    return f"EnzyGen Finished Successfully\nPredicted sequence from EnzyGen: {sequence_file}\nPredicted structure from EnzyGen: {structure_file}\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors + "\n----------\n"


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

    receptor_pdbqt = os.path.join(INPUT_DIR, "receptor_output.pdbqt")
    docked_pdbqt = os.path.join(INPUT_DIR, "docked.pdbqt")
    auto_determined = " (auto-determined)" if auto_box else ""
    
    return f"Successfully finished task. Protein pdbqt file location: {receptor_pdbqt}\nDocked ligand pdbqt file location: {docked_pdbqt}\nBinding Affinity: {binding_affinity}\nDocking score is {quality}\nDocking box is {size_x} x {size_y} x {size_z} centered at ({center_x}, {center_y}, {center_z}){auto_determined}."


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

    python_diffusion = PYTHON["diffusion"]
    command = f"{python_diffusion} {combine_protein_ligand_file} -r {INPUT_DIR}/receptor.pdbqt -l {INPUT_DIR}/ligand.pdbqt -o {INPUT_DIR}/protein_ligand_complex.pdbqt"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    command = f"conda run -n {DOCKING_ENV_NAME} obabel {INPUT_DIR}/protein_ligand_complex.pdbqt -O {INPUT_DIR}/protein_ligand_complex.pdb"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
        
    return f"Docked protein-ligand pdb file generated at: {INPUT_DIR}/protein_ligand_complex.pdb"

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
    sdf2pdb_cmd = f"conda run -n {FASTMD_CONDA_ENV} python {AGENT4MOLECULE_PATH}/mcp_agent/util/gen_ligand.py -i {MD_OUTPUT_DIR}/ligand.sdf -o {MD_OUTPUT_DIR}/ligand.pdb"
    p = subprocess.Popen(sdf2pdb_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    # Fix protein
    fixpdb_cmd = f"conda run -n {FASTMD_CONDA_ENV} python {AGENT4MOLECULE_PATH}/mcp_agent/util/protein_fix.py -i {MD_OUTPUT_DIR}/protein.pdb -o {MD_OUTPUT_DIR}/protein_fixed.pdb"
    p = subprocess.Popen(fixpdb_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

    # Merge protein and ligand 
    merge_cmd = f"conda run -n {FASTMD_CONDA_ENV} python {AGENT4MOLECULE_PATH}/mcp_agent/util/merge_complex.py -p {MD_OUTPUT_DIR}/protein_fixed.pdb -l {MD_OUTPUT_DIR}/ligand.pdb -o {MD_OUTPUT_DIR}/complex.pdb"
    p = subprocess.Popen(merge_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    
    return f"Prepared protein-ligand complex for MD at: {MD_OUTPUT_DIR}/complex.pdb and generated ligand sdf file at {MD_OUTPUT_DIR}/ligand.sdf"
    
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
    
    return f"""FastMD Simulation Completed
Output simulation files are located in the {MD_OUTPUT_DIR}/simulate_output/EnzyGen directory.

Log File:
----------
{logs}
----------

Error File:
----------
{errors}
----------
"""


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
