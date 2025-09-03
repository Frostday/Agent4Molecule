# Agent4Molecule


Query for running heme binder (as of now):


You are a scientific assistant helping to analyze protein-ligand complex diffusion simulations. A user has generated diffusion outputs using RFdiffusion and now wants to evaluate the quality of the generated backbones. Infer relevant command-line parameters for the analysis, such as the number of CPUs to use and SASA limit.


Given the input diffusion folder of /ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/input/, perform a diffusion process wit RFDiffusionAA and analyze the output. After that, use mpnn to generate predictive backbones based on the results. Create a new workspace directory with a name of 1_diffusion. The path to the script directory is /ocean/projects/cis240137p/ksubram4/Agent4Molecule/heme_binder_diffusion/
