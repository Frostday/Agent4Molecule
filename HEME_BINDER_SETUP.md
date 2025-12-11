# Setup

1. Start by cloning the Github repository outside MoleculeAgent/ project repository
```
git clone https://github.com/ikalvet/heme_binder_diffusion.git
```
2. Follow the instructions in their README to set up all the conda environments along with the RF-DiffusionAA repository - https://github.com/baker-laboratory/rf_diffusion_all_atom
3. Setting up Rosetta
    - Download Rosetta file: `wget https://downloads.rosettacommons.org/downloads/academic/3.15/rosetta_source_3.15_bundle.tar.bz2`
    - Unzip it
    - Change path for PARAMS_GENERATION variable in MoleculeAgent/mcp_agent/heme_binder_server.py
4. Next you have to change the paths at the top of MoleculeAgent/mcp_agent/heme_binder_server.py to point to the right location
5. You will also need to change slurm script creation function in this repository:
    - Inside file heme_binder_diffusion/scripts/utils/utils.py
    - Change function create_slurm_submit_script to this if you are running on PSC:
    ```
    def create_slurm_submit_script(filename, gpu=False, gres=None, time=None, mem="2g", N_nodes=1, N_cores=1, name=None, array=None, array_commandfile=None, group=None, email=None, command=None, outfile_name="output", partition="GPU-small"):
        """
        Arguments:
            time (str) :: time in 'D-HH:MM:SS'
        """
        
        # if gpu is True:
        #     assert gres is not None, "Need to specify resources when asking for a GPU"
        
        cbo = "{"
        cbc = "}"
        submit_txt = \
        f'''#!/bin/bash
    #SBATCH -N {N_nodes}
    #SBATCH -p {partition}
    #SBATCH -t {time}
    #SBATCH --gpus=v100-32:1
    #SBATCH --output={outfile_name}.log
    #SBATCH -n {N_cores}
    #SBATCH -e {outfile_name}.err
    '''
        
        if array is not None:
            if group is None:
                submit_txt += f"#SBATCH -a 1-{array}\n"
                submit_txt += f'sed -n "${cbo}SLURM_ARRAY_TASK_ID{cbc}p" {array_commandfile} | bash\n'
            else:
                N_tasks = array
                if N_tasks % group == 0:
                    N_tasks = int(N_tasks / group)
                else:
                    N_tasks = int(N_tasks // group) + 1
                submit_txt += f'#SBATCH -a 1-{N_tasks}\n'
                submit_txt += f"GROUP_SIZE={group}\n"
            
                submit_txt += "LINES=$(seq -s 'p;' $((($SLURM_ARRAY_TASK_ID-1)*$GROUP_SIZE+1)) $(($SLURM_ARRAY_TASK_ID*$GROUP_SIZE)))\n"
                submit_txt += f'sed -n "${cbo}LINES{cbc}p" ' + f"{array_commandfile} | bash -x\n"
        else:
            submit_txt += f"\n{command}\n"
        
        with open(filename, "w") as file:
            for l in submit_txt:
                file.write(l)
    ```
    - You will also need to adapt the function calls inside MoleculeAgent/mcp_agent/heme_binder_server.py if you are not running on PSC.
6. Inside file heme_binder_diffusion/scripts/design/scoring/heme_scoring.py - comment out filters and align_atoms
7. (If required) Fixing the GLIBC errors
    - Download GLIBC libraries from here - https://drive.google.com/file/d/1DvODchL0ImHnBcYfW33qRM4bhtTyIY-E/view?usp=sharing and unzip
    - Create a new file at heme_binder_diffusion/scripts/utils/DAlphaBall.sh (make sure to change the paths)
    ```
    #!/bin/bash

    GLIBC=$HOME/glibc_install/lib
    GMP=$HOME/gmp_install/lib
    GFORTRAN=/ocean/projects/cis240137p/dgarg2/miniconda3/envs/diffusion/lib

    export LD_LIBRARY_PATH=$GLIBC:$GMP:$GFORTRAN:$LD_LIBRARY_PATH

    $GLIBC/ld-2.29.so \
        --library-path $GLIBC:$GMP:$GFORTRAN \
        /ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/scripts/utils/DAlphaBall.gcc "$@"
    ```
    - Change `DAB = f"{SCRIPT_PATH}/../utils/DAlphaBall.gcc"` to `DAB = f"{SCRIPT_PATH}/../utils/DAlphaBall.sh"` inside all of these files:
        - heme_binder_diffusion/scripts/diffusion_analysis/process_diffusion_outputs.py
        - heme_binder_diffusion/scripts/design/heme_pocket_ligMPNN.py
        - heme_binder_diffusion/scripts/design/heme_pocket_FastDesign.py
        - heme_binder_diffusion/scripts/design/align_add_ligand_relax.py
8. (Testing) For testing individual tools, comment `mcp.run(transport='stdio')` and call the functions inside the file with the right arguments.
