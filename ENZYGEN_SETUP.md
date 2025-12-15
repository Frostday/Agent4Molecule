# EnzyGen MCP Server Setup Documentation

This guide provides comprehensive setup instructions for the EnzyGen MCP Server, including all required dependencies, conda environments, and path configurations.

## Table of Contents
1. [EnzyGen Setup](#1-enzygen-setup)
2. [Docking Environment (AutoDock Vina)](#2-docking-environment-autodock-vina)
3. [ESP Prediction Environment](#3-esp-prediction-environment)
4. [FastMD Simulation Setup](#4-fastmd-simulation-setup)
5. [ColabFold Setup](#5-colabfold-setup)
6. [Path Configuration](#6-path-configuration)
7. [Environment Variable Summary](#7-environment-variable-summary)
8. [Testing the Setup](#8-testing-the-setup)
9. [Additional Utility Scripts](#9-additional-utility-scripts)

---

## Prerequisites

- Conda/Anaconda installed
- Access to GPU (for AlphaFold2/ColabFold and MD simulations)
- Git
- Python 3.8+

---

## 1. EnzyGen Setup

### Clone the Repository
```bash
# Clone outside your main project directory
cd /path/to/your/github/folder
git clone https://github.com/LeiLiLab/EnzyGen.git
cd EnzyGen
```

### Create EnzyGen Conda Environment
Follow the instructions in the EnzyGen README to set up the conda environment. Typically:

```bash
conda create -n enzygen python=3.8 -y 
conda activate enzygen 
pip install setuptools==50.0
python -m pip install "pip<24.1"
pip install setuptools==50.0
pip install cffi
pip install omegaconf==2.0.6
pip install numpy==1.23.5
pip install scikit-learn==1.3.2
pip install importlib_resources==5.12.0
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
bash setup.sh
```

### Note the Python Path
After activation, note your Python path:
```bash
which python
```

---

## 2. Docking Environment (AutoDock Vina)
### Setup autodocking vida conda environment

```
$ conda create -n docking python=3.11
$ conda activate docking
$ conda config --env --add channels conda-forge
```

```
$ conda install -c conda-forge numpy swig boost-cpp libboost sphinx sphinx_rtd_theme
$ pip install vina
```

### Download the vina executable from github
Github link [https://github.com/ccsb-scripps/AutoDock-Vina/releases]

```
$ wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64
$ chmod +x vina_1.2.7_linux_x86_64
$ mv vina_1.2.7_linux_x86_64 $CONDA_PREFIX/bin/vina
$ which vina
$ vina --version
```

### Download meeko
```
pip install -U scipy rdkit meeko gemmi prody
```

### Install Additional Docking Tools
```bash
# Install OpenBabel for file format conversion
conda install -c conda-forge openbabel

# Install RDKit
conda install -c conda-forge rdkit
```

### Verify Installation
```bash
vina --help
mk_prepare_receptor.py --help
mk_prepare_ligand.py --help
```

---

## 3. ESP Prediction Environment

### Clone ESP Prediction Repository
```bash
cd /path/to/your/github/folder
git clone https://github.com/AlexanderKroll/ESP_prediction_function.git
```

### Create ESP Environment
```bash
conda create -n esp -c conda-forge pandas==1.3.1 python=3.8 jupyter  numpy==1.23.1 fair-esm==0.4.0 py-xgboost=1.3.3 rdkit=2022.09.5
conda activate esp
conda remove py-xgboost
pip install xgboost
```

---

## 4. FastMD Simulation Setup

### Clone FastMDSimulation Repository
```bash
cd /path/to/your/github/folder
git clone https://github.com/aai-research-lab/FastMDSimulation.git
cd FastMDSimulation
```

### Create FastMD Conda Environment
```bash
conda env create -f environment.gpu.yml
conda activate fastmds
```

### Verify Installation
```bash
fastmds --help
```

### Modify FastMD Code for Ligand Support

To enable ligand parameterization with OpenFF, edit `openmm_engine.py` in the FastMD installation:

1. Locate the `_build_simulation` function
2. Add the following code block after the force field initialization:

```python
# --- Add OpenFF ligand parameterization ---
ligand_file = defaults.get("ligand_file")
if ligand_file:
    from openff.toolkit.topology import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    ligand = Molecule.from_file(str(ligand_file))
    openff_version = defaults.get("openff_forcefield", "openff-2.2.0.offxml")
    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=[ligand],
        forcefield=openff_version,
    )

    ff.registerTemplateGenerator(smirnoff.generator)
    logger.info(f"Registered OpenFF ligand template from {ligand_file}")
# ------------------------------------------------------------------------
```

This modification allows FastMD to properly parameterize small molecule ligands using the OpenFF force field.

Make sure to use "amber14-all.xml", "amber14/tip3pfb.xml" forcefield when running the simulation. 

---

## 5. ColabFold Setup

### Allocate GPU Node (if on HPC cluster)
```bash
srun --partition=GPU-shared --gres=gpu:1 --time=1:00:00 --pty bash
```

### Create Directory Structure
```bash
# Set your base path (adjust to your system)
export BASE_PATH="/your/path/to"

mkdir -p $BASE_PATH/colabfold
cd $BASE_PATH/colabfold

# Create Apptainer cache directories
mkdir -p $BASE_PATH/apptainer_cache
mkdir -p $BASE_PATH/apptainer_tmp

export APPTAINER_CACHEDIR=$BASE_PATH/apptainer_cache
export APPTAINER_TMPDIR=$BASE_PATH/apptainer_tmp

# Create ColabFold directories
mkdir -p $BASE_PATH/colabfold/cf_cache
mkdir -p $BASE_PATH/colabfold/cf_out
mkdir -p $BASE_PATH/colabfold/cf_toy
```

### Download ColabFold Container
```bash
# Clean cache and pull the Singularity/Apptainer image
apptainer cache clean -a
apptainer pull colabfold_1.5.5-cuda12.2.2.sif docker://ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2
```

### Download AlphaFold2 Model Weights
```bash
apptainer exec --nv \
  -B $BASE_PATH/colabfold/cf_cache:/cache \
  colabfold_1.5.5-cuda12.2.2.sif \
  python -m colabfold.download
```

### Test ColabFold Installation

1. **Create a test sequence file:**
```bash
mkdir -p $BASE_PATH/EnzyGen/af2_outputs/
cat > $BASE_PATH/EnzyGen/af2_outputs/input_seq.fasta << 'EOF'
>enzygen
SHMRPEPRLITILFSDIVGFTRMSNALQSQGVAELLNEYLGEMTRAVFENQGTVDKFVGDAIMALYGAPEEMSPSEQVRRAIATARQMLVALEKLNQGWQERGLVGRNEVPPVRFRCGIHQGMAVVGLFGSQERSDFTAIGPSVNIAARLQEATAPNSIMVSAMVAQYVPDEEIIKREFLELKGIDEPVMTCVINPNM
EOF
```

2. **Run ColabFold prediction:**
```bash
apptainer exec --nv \
  -B $BASE_PATH/colabfold/cf_cache:/cache \
  -B $BASE_PATH/EnzyGen/af2_outputs:/work \
  $BASE_PATH/colabfold/colabfold_1.5.5-cuda12.2.2.sif \
  colabfold_batch /work/input_seq.fasta /work/out \
    --msa-mode mmseqs2_uniref_env \
    --pair-mode unpaired_paired \
    --use-gpu-relax \
    --num-seeds 1 \
    --num-models 1 \
    --model-type alphafold2_ptm
```

**Expected output:** PDB structure files in `$BASE_PATH/EnzyGen/af2_outputs/out/`

---

## 6. Path Configuration

### Update `enzygen_server.py` Paths

Edit `/path/to/MoleculeAgent/src/mcp_agent/enzygen_server.py` and update the following variables:

```python
# Path to EnzyGen repository
ENZYGEN_PATH = "/your/path/to/EnzyGen"

# Path to EnzyGen conda environment Python
ENZYGEN_CONDA_ENV = "/your/conda/envs/enzygen/bin/python"

# ColabFold paths
COLABFOLD_CACHE = "/your/path/to/colabfold/cf_cache"
COLABFOLD_SIF = "/your/path/to/colabfold/colabfold_1.5.5-cuda12.2.2.sif"

# Utility script path
combine_protein_ligand_file = "/your/path/to/MoleculeAgent/mcp_agent/util/combine_protein_ligand.py"

# Python environments
PYTHON = {
    "diffusion": "/your/conda/envs/diffusion/bin/python",  # Optional
    "vina": "/your/conda/envs/docking/bin/python"
}

# Conda environment names
DOCKING_ENV_NAME = "docking"
ESP_CONDA_ENV = "esp"
FASTMD_CONDA_ENV = "fastmds"

# Paths to other repositories
FASTMD_PATH = "/your/path/to/FastMDSimulation"
MOLECULE_AGENT_PATH = "/your/path/to/MoleculeAgent/"

# Working directory (this can be customized per run)
EC_FOLDER = "2.4.1.135"  # Default EC number for output organization
```

---

## 7. Environment Variable Summary

### Required Conda Environments

| Environment | Purpose | Key Packages |
|------------|---------|--------------|
| `enzygen` | Enzyme generation | fairseq, pytorch, biopython |
| `docking` | Molecular docking | autodock-vina, meeko, openbabel, rdkit |
| `esp` | ESP scoring | xgboost, pandas, numpy, rdkit |
| `fastmds` | MD simulations | openmm, pyyaml, pdbfixer |

---

## 8. Testing the Setup

### Test MCP Server
```bash
cd /path/to/MoleculeAgent/src/mcp_agent

# Run server with client (requires main conda environment)
python client.py enzygen_server.py
```

**Note:** The server should start without errors. Test individual tools through the MCP interface.

---

## 9. Additional Utility Scripts

The following utility scripts should be present in `/path/to/MoleculeAgent/mcp_agent/util/`:

- `combine_protein_ligand.py` - Combines protein and ligand PDBQT files
- `mol_to_sdf.py` - Converts MOL files to SDF format
- `clean_fragment.py` - Cleans SDF files
- `protein_fix.py` - Fixes PDB files for MD
- `merge_complex.py` - Merges protein and ligand PDB files
- `gen_ligand.py` - Generates ligand PDB from SDF
- `msa_to_motif.py` - Converts MSA to motif format

---

## Support

For issues with specific tools:
- EnzyGen: [GitHub Issues](https://github.com/LeiLiLab/EnzyGen/issues)
- AutoDock Vina: [Documentation](https://autodock-vina.readthedocs.io/)
- ESP: [GitHub Issues](https://github.com/AlexanderKroll/ESP_prediction_function/issues)
- FastMD: [GitHub Issues](https://github.com/aai-research-lab/FastMDSimulation/issues)
- ColabFold: [GitHub Issues](https://github.com/sokrypton/ColabFold/issues)
