# Setup

1. Start by cloning the Github repository outside MoleculeAgent/ project repository
```
git clone https://github.com/JocelynSong/PPDiff.git
```
2. Follow the instructions in their README to set up the environments, datasets and models
3. Set up the AlphaFold3 repository (https://github.com/google-deepmind/alphafold3) outside PPDiff/
    - Follow the instructions in the repository till you reach the docker setup (get access to the datasets and models weights)
    - We will use apptainer instead of docker so you can directly pull a prebuilt image using - `apptainer pull alphafold3.sif docker://baldikacti/alphafold3:latest`
    - Then convert it to an apptainer sif using - `apptainer exec --nv alphafold3.sif nvidia-smi`
4. Next you have to change the paths at the top of MoleculeAgent/src/mcp_agent/ppdiff.py to point to the right location
