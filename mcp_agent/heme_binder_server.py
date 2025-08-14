import os
import glob
from typing import Annotated
from pydantic import Field
import json
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt

from run_utils import extract_job_id, RF_DIFFUSION_CONFIG

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
    "general": f"{CONDAPATH}/envs/diffusion/bin/python"
}
if not os.path.exists(WDIR):
    os.makedirs(WDIR, exist_ok=True)

# INPUTS
# params = [f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"]
# LIGAND = "HBA"
# diffusion_inputs = ['/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb']


@mcp.tool()
def run_rf_diffusion(
    input_pdb: Annotated[str, Field(description="Input PDB file")],
    ligand: Annotated[str, Field(description="Ligand name")],
    N_designs: Annotated[int, Field(description="Number of designs to generate")] = 5,
    T_steps: Annotated[int, Field(description="Number of diffusion steps")] = 200,
    job_time: Annotated[str, Field(description="Time limit for diffusion jobs (estimated time is 3.5 * T_steps * N_design seconds)")] = "1:00:00"
) -> str:
    DIFFUSION_DIR = f"{WDIR}/0_diffusion"
    if not os.path.exists(DIFFUSION_DIR):
        os.makedirs(DIFFUSION_DIR, exist_ok=False)
    os.chdir(DIFFUSION_DIR)
    config = RF_DIFFUSION_CONFIG.format(
        T_steps=T_steps,
        N_designs=N_designs,
        LIGAND=ligand
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
        diffusion_outputs = glob.glob(f"{rundir}/out/*.pdb")
        diffusion_output_files += diffusion_outputs
        for out in diffusion_outputs:
            with open(out, "r") as f:
                output_preds.append(f"File: {os.path.join(DIFFUSION_DIR, out)}\n\n" + f.read())
        with open(os.path.join(rundir, "output.log"), "r") as f:
            rfa_logs.append(f"File: {os.path.join(DIFFUSION_DIR, rundir, 'output.log')}\n\n" + f.read())
            
    return f"Predicted backbone structure(s) from RFDiffusionAA:\n\nAll Output Files: {diffusion_output_files}\n\n----------\n" + "\n----------\n----------\n".join(output_preds) + "\n----------\n\n" + "Log File:\n\n----------\n" + logs + "\n----------\n\n" + "RFDiffusionAA Log Files:\n\n----------\n" + "\n----------\n----------\n".join(rfa_logs) + "\n----------\n\n" + "Error File:\n\n----------\n" + errors


@mcp.tool()
def analyze_rf_diffusion_outputs(
    diffusion_outputs: Annotated[list[str], Field(description="List of diffusion output files")],
    ref_pdb: Annotated[str, Field(description="Reference PDB file (same as input PDB file)")],
    params: Annotated[list[str], Field(description="Rosetta params file(s)")],
    term_limit: Annotated[float, Field(description="Terminal residue limit")] = None,
    SASA_limit: Annotated[float, Field(description="Solvent accessible surface area limit")] = None,
    loop_limit: Annotated[float, Field(description="Maximum fraction of backbone that can be in loop conformation")] = None,
    longest_helix: Annotated[int, Field(description="Maximum length of helices")] = None,
    rog: Annotated[float, Field(description="Radius of gyration limit for protein compactness")] = None,
    ref_catres: Annotated[str, Field(description="Position of CYS in diffusion input")] = None,
    exclude_clash_atoms: Annotated[str, Field(description="Ligand atoms excluded from clashchecking because they are flexible")] = None,
    ligand_exposed_atoms: Annotated[str, Field(description="Ligand atoms that need to be more exposed")] = None,
    exposed_atom_SASA: Annotated[float, Field(description="Minimum absolute SASA for exposed ligand atoms")] = None,
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
    if ligand_exposed_atoms is not None:
        dif_analysis_cmd_dict["--ligand_exposed_atoms"] = ligand_exposed_atoms
    if exposed_atom_SASA is not None:
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

    return f"Number of good backbone structure(s): {len(diffused_backbones_good)}\nGood backbone structure(s) files: {diffused_backbones_good}\n\nDistribution plot path: {img_path}\n\nAnalysis CSV:\n\n----------\n" + dif_analysis_df.to_csv(index=False) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors


@mcp.tool()
def run_protein_mpnn(
    diffused_backbones_good: Annotated[list[str], Field(description="List of good diffused backbone PDBs")],
    MPNN_temperatures: Annotated[list, Field(description="List of temperatures for ProteinMPNN")] = [0.1, 0.2, 0.3],
    MPNN_outputs_per_temperature: Annotated[int, Field(description="Number of outputs per temperature")] = 5,
    MPNN_omit_AAs: Annotated[str, Field(description="Amino acids to omit")] = "CM",
    job_time: Annotated[str, Field(description="Time limit for ProteinMPNN jobs")] = "1:00:00"
):
    MPNN_DIR = f"{WDIR}/1_proteinmpnn"
    assert len(diffused_backbones_good) > 0, "No good backbones found!"
    os.makedirs(MPNN_DIR, exist_ok=True)
    os.chdir(MPNN_DIR)

    mask_json_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/design/make_maskdict_from_trb.py --out masked_pos.jsonl --trb"
    for d in diffused_backbones_good:
        mask_json_cmd += " " + d.replace(".pdb", ".trb")
    p = subprocess.Popen(mask_json_cmd, shell=True)
    (output, err) = p.communicate()
    assert os.path.exists("masked_pos.jsonl"), "Failed to create masked positions JSONL file"

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
    
    # return f"Predicted structure(s) from ProteinMPNN:\n\nAll Output Files: {protein_seq_fasta_files}\n\nSequences:\n\n----------\n" + "\n----------\n----------\n".join(sequences) + "\n----------\n\nBackbones:\n\n----------\n" + "\n----------\n----------\n".join(backbones) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors
    return f"Predicted sequence(s) from ProteinMPNN:\n\nAll Output Files: {protein_seq_fasta_files}\n\n----------\n" + "\n----------\n----------\n".join(sequences) + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors


def run_af2(
    protein_seq_fasta_files: Annotated[list[str], Field(description="Protein sequence fasta files (ProteinMPNN outputs)")],
    AF2_recycles: Annotated[int, Field(description="Number of AF2 recycling steps")] = 3,
    AF2_models: Annotated[str, Field(description="Number of AF2 models (default is '4', add other models to this string if needed, i.e. '3 4 5')")] = "4",
    job_time: Annotated[str, Field(description="Time limit for AF2 jobs")] = "1:00:00"
):
    AF2_DIR = f"{WDIR}/2_af2"
    os.makedirs(AF2_DIR, exist_ok=True)
    os.chdir(AF2_DIR)

    # First collecting MPNN outputs and creating FASTA files for AF2 input
    mpnn_fasta = utils.parse_fasta_files(protein_seq_fasta_files)
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
    with open(glob.glob(f"{AF2_DIR}/*.csv")[0], "r") as f:
        afs_scores_csv = f.read()
    af2_out_files = glob.glob(f"{AF2_DIR}/*.pdb")
    
    # return f"Predicted structure(s) from AF2:\n\nAll Output Files: {af2_out_files}\n\n----------\n" + "\n----------\n----------\n".join(structures) + "\n----------\n\nInput Sequences:\n\n----------\n" + input_sequences + "\n----------\n\nAF2 Scores CSV:\n\n----------\n" + afs_scores_csv + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors
    return f"Predicted structure(s) from AF2:\n\nAll Output Files: {af2_out_files}\n\n----------\n" + "\n----------\n----------\n".join(structures) + "\n----------\n\nAF2 Scores CSV:\n\n----------\n" + afs_scores_csv + "\n----------\n\nLog File:\n\n----------\n" + logs + "\n----------\n\nError File:\n\n----------\n" + errors


def analyze_af2_outputs():
    return


def cleanup():
    os.system(f"rm -rf /ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/*")
    os.system(f"rm -rf {WDIR}/*")
    # os.system(f"rm -rf {WDIR}/0_diffusion/*")
    # os.system(f"rm -rf {WDIR}/1_proteinmpnn/*")
    # os.system(f"rm -rf {WDIR}/2_af2/*")


if __name__ == "__main__":
    # mcp.run(transport='stdio')

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
    #     ref_pdb="/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb", 
    #     params=[f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"],
    #     term_limit=15.0,
    #     SASA_limit=0.3,
    #     loop_limit=0.4,
    #     longest_helix=30,
    #     rog=30.0,
    #     ref_catres="A15",
    #     exclude_clash_atoms="O1 O2 O3 O4 C5 C10",
    #     ligand_exposed_atoms="C45 C46 C47",
    #     exposed_atom_SASA=10.0
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
    #     AF2_recycles=3,
    #     AF2_models="4",
    #     job_time="1:00:00"
    # )
    # with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_2.txt", "w") as f:
    #     f.write(output)

    output = analyze_af2_outputs()
    with open("/ocean/projects/cis240137p/dgarg2/github/Agent4Molecule/mcp_agent/outputs/output_2_analyze.txt", "w") as f:
        f.write(output)
