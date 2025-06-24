from mcp.server.fastmcp import FastMCP
from typing import Any,Annotated,Optional
import httpx
import os,sys,glob
from pydantic import Field
sys.path.append("/ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/scripts/utils")
import utils,json
from google import genai
import getpass
import subprocess
import time
import importlib
from shutil import copy2
from google.genai.types import Content, Part



mcp = FastMCP("heme-binder")


@mcp.tool()
def perform_diffusion(script_dir: Annotated[str, Field(description="Location of script directory")],
                      work_dir: Annotated[str, Field(description="Workspace for jobs and analysis")],
     
                    diffusion_folder: Annotated[str, Field(description="Location of PDB files")]) -> str:
    SCRIPT_DIR = os.path.dirname(script_dir)  # edit this to the GitHub repo path. Throws an error by default.
    if not os.path.exists(SCRIPT_DIR):
        raise RuntimeError(f"Missing SCRIPT_DIR: {SCRIPT_DIR}")
    sys.path.append(SCRIPT_DIR+"/scripts/utils")
    diffusion_script = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/rf_diffusion_all_atom/run_inference.py"
    proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"

    CONDAPATH = "/ocean/projects/cis240137p/ksubram4/anaconda3"   # edit this depending on where your Conda environments live
    PYTHON = {"diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
          "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
          "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
          "general": f"{CONDAPATH}/envs/diffusion/bin/python"}
    
    WDIR = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/outputs"

    if not os.path.exists(WDIR):
        os.makedirs(WDIR, exist_ok=True)

    print(f"Working directory: {WDIR}",file=sys.stderr)

    USE_GPU_for_AF2 = True
    params = [f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"]  # Rosetta params file(s)
    LIGAND = "HBA"

    diffusion_inputs = glob.glob(diffusion_folder + "*.pdb")
    print(f"Found {len(diffusion_inputs)} PDB files",file=sys.stderr)


    DIFFUSION_DIR = f"{WDIR}/" + work_dir
    if not os.path.exists(DIFFUSION_DIR):
        os.makedirs(DIFFUSION_DIR, exist_ok=False)

    os.chdir(DIFFUSION_DIR)
    N_designs = 5
    T_steps = 200

## Edit this config based on motif residues, etc...
    config = f"""
defaults:
  - aa

diffuser:
  T: {T_steps}

inference:
  num_designs: {N_designs}
  model_runner: NRBStyleSelfCond
  ligand: '{LIGAND}'

model:
  freeze_track_motif: True

contigmap:
  contigs: ["30-110,A15-15,30-110"]
  inpaint_str: null
  length: "100-140"

potentials:
  guiding_potentials: ["type:ligand_ncontacts,weight:1"] 
  guide_scale: 2
  guide_decay: cubic
"""

    estimated_time = 3.5 * T_steps * N_designs  # assuming 3.5 seconds per timestep on A4000 GPU

    print(f"Estimated time to produce {N_designs} designs = {estimated_time/60:.0f} minutes",file=sys.stderr)
    with open("config.yaml", "w") as file:
        file.write(config)
    print(f"Wrote config file to {os.path.realpath('config.yaml')}",file=sys.stderr)

    print("Done",file=sys.stderr)
    return "Sucessfully completed step."


@mcp.tool()
def analyze_diffusion_output(script_dir: Annotated[str, Field(description="File provided for user")],
                             work_dir: Annotated[str, Field(description="Workspace for jobs and analysis")]):
    SCRIPT_DIR = os.path.dirname(script_dir)  # edit this to the GitHub repo path. Throws an error by default.
    if not os.path.exists(SCRIPT_DIR):
        raise RuntimeError(f"Missing SCRIPT_DIR: {SCRIPT_DIR}")
    sys.path.append(SCRIPT_DIR+"/scripts/utils")
    diffusion_script = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/rf_diffusion_all_atom/run_inference.py"
    proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"

    CONDAPATH = "/ocean/projects/cis240137p/ksubram4/anaconda3"   # edit this depending on where your Conda environments live
    PYTHON = {"diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
          "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
          "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
          "general": f"{CONDAPATH}/envs/diffusion/bin/python"}
    
    WDIR = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/outputs"

    if not os.path.exists(WDIR):
        os.makedirs(WDIR, exist_ok=True)

    print(f"Working directory: {WDIR}")

    USE_GPU_for_AF2 = True
    params = [f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"]  # Rosetta params file(s)
    LIGAND = "HBA"

    diffusion_inputs = glob.glob(f"{SCRIPT_DIR}/input/*.pdb")
    print(f"Found {len(diffusion_inputs)} PDB files")


    # DIFFUSION_DIR = f"{WDIR}/0_diffusion"
    DIFFUSION_DIR = f"{WDIR}/" + work_dir
    if not os.path.exists(DIFFUSION_DIR):
        os.makedirs(DIFFUSION_DIR, exist_ok=False)

    os.chdir(DIFFUSION_DIR)

    diffusion_rundirs = ['7o2g_HBA']
    analysis_script = f"{SCRIPT_DIR}/scripts/diffusion_analysis/process_diffusion_outputs.py"

    diffusion_outputs = []
    for d in diffusion_rundirs:
        diffusion_outputs += glob.glob(f"{d}/out/*.pdb")
    
    analysis_script = f"{SCRIPT_DIR}/scripts/diffusion_analysis/process_diffusion_outputs.py"
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


## Creating a Slurm submit script
## adjust time depending on number of designs and available hardware
    submit_script = "submit_diffusion.sh"
    utils.create_slurm_submit_script(filename=submit_script, name="diffusion_example", gpu=True, gres="gpu:a4000:1",
                                 mem="8g", N_cores=2, time="1:00:00", email="kaavi.subu@gmail.com",
                                 array=len(commands_diffusion), array_commandfile=cmds_filename)

    print(f"Writing diffusion submission script to {submit_script}")
    print(f"{len(commands_diffusion)} diffusion jobs to run")

    if not os.path.exists(DIFFUSION_DIR+"/.done"):
        p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()



# By default I don't use the --analyze flag. As a result the backbones are filtered as the script runs.
# You can set --analyze to True to calculate all scores for all backbones.
# This will slow the analysis down, but you can then filter the backbones separately afterwards.
    dif_analysis_cmd_dict = {"--pdb": " ".join(diffusion_outputs),
                        # "--ref": f"{SCRIPT_DIR}/input/*.pdb",
                        "--ref": f"{SCRIPT_DIR}/input/7o2g_HBA.pdb",
                        "--params": " ".join(params),
                        "--term_limit": "15.0",
                        "--SASA_limit": "0.3",  # Highest allowed relative SASA of ligand
                        "--loop_limit": "0.4",  # Fraction of backbone that can be loopy
                        "--ref_catres": "A15",  # Position of CYS in diffusion input
                        "--rethread": True,
                        "--fix": True,
                        "--exclude_clash_atoms": "O1 O2 O3 O4 C5 C10",  # Ligand atoms excluded from clashchecking because they are flexible
                        "--ligand_exposed_atoms": "C45 C46 C47",  # Ligand atoms that need to be more exposed
                        "--exposed_atom_SASA": "10.0",  # minimum absolute SASA for exposed ligand atoms
                        "--longest_helix": "30",
                        "--rog": "30.0",
                        "--partial": None,
                        "--outdir": None,
                        "--traj": "5/30",  # Also random 5 models are taken from the last 30 steps of the diffusion trajectory
                        "--trb": None,
                        "--analyze": False,
                        "--nproc": "2"}

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
            print(k, val)

    # print(f"Analysis command: {analysis_command}")

    if len(diffusion_outputs) < 100:
    ## Analyzing locally
        p = subprocess.Popen(analysis_command, shell=True)
        (output, err) = p.communicate()
    else:
    ## Too many structures to analyze.
    ## Running the analysis as a SLURM job.
        submit_script = "submit_diffusion_analysis.sh"
        utils.create_slurm_submit_script(filename=submit_script, name="diffusion_analysis",
                                     mem="8g", N_cores=dif_analysis_cmd_dict["--nproc"], time="0:20:00", email=EMAIL,
                                     command=analysis_command, outfile_name="output_analysis")

        p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
    return "Succesfully finished task."


@mcp.tool()
async def run_alphafold(script_dir: Annotated[str, Field(description="File provided for user")],
                             work_dir: Annotated[str, Field(description="Workspace for jobs and analysis")]) -> str:

    SCRIPT_DIR = os.path.dirname(script_dir)  # edit this to the GitHub repo path. Throws an error by default.
    if not os.path.exists(SCRIPT_DIR):
        raise RuntimeError(f"Missing SCRIPT_DIR: {SCRIPT_DIR}")
    sys.path.append(SCRIPT_DIR+"/scripts/utils")
    diffusion_script = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/rf_diffusion_all_atom/run_inference.py"
    proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"

    CONDAPATH = "/ocean/projects/cis240137p/ksubram4/anaconda3"   # edit this depending on where your Conda environments live
    PYTHON = {"diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
          "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
          "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
          "general": f"{CONDAPATH}/envs/diffusion/bin/python"}
    
    WDIR = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/outputs"

    if not os.path.exists(WDIR):
        os.makedirs(WDIR, exist_ok=True)

    print(f"Working directory: {WDIR}")

    USE_GPU_for_AF2 = True
    params = [f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"]  # Rosetta params file(s)
    LIGAND = "HBA"



@mcp.tool()
async def infer_analysis_params(
    text: Annotated[str, Field(description="A short task description")],
    context: Annotated[str, Field(description="Optional context from previous steps or output stats")] = ""
) -> dict:
    """
    Infers analysis command parameters for diffusion output when user provides minimal input.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = f"""
You are a molecular modeling assistant.

Given a user task and some context (e.g., number of structures, ligand name, diffusion quality), suggest command-line parameters for analysis.

Return JSON with the following keys. If the user specifies the value of one of these keys that value must be used. Otherwise, decide if it is best to use the default parameter values or come up with your own values based on the task and context.
--SASA_limit, --loop_limit, --term_limit, --nproc, --analyze, --ref_catres, ----rethread,--fix,--exclude_clash_atoms,-ligand_exposed_atoms,-exposed_atom_SASA,--longest_helix,--rog,--partial,--traj,--trb



Only include parameters that are relevant. Pick reasonable values even if not specified by the user.

TASK:
{text}

CONTEXT:
{context}

JSON:
    """

    try:
        # print("starting tool",file=sys.stderr)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[Content(role="user", parts=[Part.from_text(text=prompt)])]
        )
        # print("got response",file=sys.stderr)
        raw = response.candidates[0].content.parts[0].text.strip()
        # print(raw,file=sys.stderr)
        return json.loads(raw[7:-4])
    except Exception as e:
        return {"error": f"Failed to generate parameters: {str(e)}"}


@mcp.tool()
def run_mpnn(script_dir: Annotated[str, Field(description="File provided for user")],
                             work_dir: Annotated[str, Field(description="Workspace for jobs and analysis")]):
    SCRIPT_DIR = os.path.dirname(script_dir)  # edit this to the GitHub repo path. Throws an error by default.
    if not os.path.exists(SCRIPT_DIR):
        raise RuntimeError(f"Missing SCRIPT_DIR: {SCRIPT_DIR}")
    sys.path.append(SCRIPT_DIR+"/scripts/utils")
    diffusion_script = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/rf_diffusion_all_atom/run_inference.py"
    proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"

    CONDAPATH = "/ocean/projects/cis240137p/ksubram4/anaconda3"   # edit this depending on where your Conda environments live
    PYTHON = {"diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
          "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
          "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
          "general": f"{CONDAPATH}/envs/diffusion/bin/python"}
    
    WDIR = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/outputs"

    if not os.path.exists(WDIR):
        os.makedirs(WDIR, exist_ok=True)

    # print(f"Working directory: {WDIR}")

    USE_GPU_for_AF2 = True
    params = [f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"]  # Rosetta params file(s)
    LIGAND = "HBA"

    diffusion_inputs = glob.glob(f"{SCRIPT_DIR}/input/*.pdb")
    print(f"Found {len(diffusion_inputs)} PDB files")


    # DIFFUSION_DIR = f"{WDIR}/0_diffusion"
    DIFFUSION_DIR = f"{WDIR}/" + work_dir
    
    diffused_backbones_good = glob.glob(f"{DIFFUSION_DIR}/filtered_structures/*.pdb")
    # assert len(diffused_backbones_good) > 0, "No good backbones found!"

    os.chdir(WDIR)
    MPNN_DIR = f"{WDIR}/1_proteinmpnn"
    os.makedirs(MPNN_DIR, exist_ok=True)
    os.chdir(MPNN_DIR)

### Parsing diffusion output TRB files to extract fixed motif residues
## These residues will not be redesigned with proteinMPNN
    mask_json_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/design/make_maskdict_from_trb.py --out masked_pos.jsonl --trb"
    for d in diffused_backbones_good:
        mask_json_cmd += " " + d.replace(".pdb", ".trb")

    p = subprocess.Popen(mask_json_cmd, shell=True)
    (output, err) = p.communicate()



### Setting up proteinMPNN run commands
## We're doing design with 3 temperatures, and 5 sequences each.
## This usually gives decent success with designable backbones.
## For more complicated cases consider doing >100 sequences.

    MPNN_temperatures = [0.1, 0.2, 0.3]
    MPNN_outputs_per_temperature = 5
    MPNN_omit_AAs = "CM"

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

    # print("Example MPNN command:")
    print(commands_mpnn[-1])

### Running proteinMPNN with Slurm.
### Grouping jobs with 10 commands per one array job.

    submit_script = "submit_mpnn.sh"
    utils.create_slurm_submit_script(filename=submit_script, name="1_proteinmpnn", mem="4g", 
                                 N_cores=2, time="1:00:00", email="kaavi.subu@gmail.com", array=len(commands_mpnn),
                                 array_commandfile=cmds_filename_mpnn, group=10)

    if not os.path.exists(MPNN_DIR+"/.done"):
        p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()

    return "Successfully finished task."




if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')