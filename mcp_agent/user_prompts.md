### Prompt for enzygen:

1. Design an enzyme using the given data:\\n- EC4 category = \"4.6.1.1\"\\n- Information about the motif:\\n    - Amino acid = \"I\", Coordinates = [1.0, 1.0, 1.0], Index = 1\\n    - Amino acid = \"G\", Coordinates = [2.0, 1.8, 1.5], Index = 4\\n    - Amino acid = \"D\", Coordinates = [1.1, 1.2, 2.0], Index = 0\\n- PDB file = \"5cxl.A\"\\n- Recommended length = 20

<!-- Design an enzyme using the given data: 
- EC4 category = "4.6.1.1" 
- Information about the motif:
    - Amino acid = "I", Coordinates = [1.0, 1.0, 1.0], Index = 1
    - Amino acid = "G", Coordinates = [2.0, 1.8, 1.5], Index = 4
    - Amino acid = "D", Coordinates = [1.1, 1.2, 2.0], Index = 0
- PDB file = "5cxl.A" 
- Recommended length = 20  -->

1. Design an enzyme that functions as an adenylate-processing protein, acting like a cyclase to transform ATP into 3’,5’-cyclic AMP while releasing pyrophosphate. The enzyme should resemble known adenylylcyclases in structure and activity, and be capable of catalyzing the formation of cyclic AMP as a signaling molecule.

2. Design a novel adenylate cyclase enzyme molecule that catalyzes the ATP diphosphate-lyase (cyclizing) reaction, converting ATP into 3’,5’-cyclic AMP. The structure should resemble natural adenylylcyclases while maintaining efficient 3’,5’-cyclic AMP synthetase activity.

3. Construct the molecular blueprint of an adenyl cyclase protein capable of acting as a 3’,5’-cyclic AMP synthetase. The enzyme must carry out the ATP diphosphate-lyase (cyclizing) step, producing 3’,5’-cyclic AMP from ATP in a manner similar to known adenylylcyclases.

4. Generate an engineered adenylylcyclase enzyme molecule optimized for strong signaling output. It should catalyze ATP conversion into 3’,5’-cyclic AMP via its ATP diphosphate-lyase (cyclizing) function and demonstrate features characteristic of natural adenylate cyclases.

5. Propose the design of a synthetic adenyl cyclase protein molecule that mimics a 3’,5’-cyclic AMP synthetase. The enzyme should efficiently transform ATP into 3’,5’-cyclic AMP while operating through an ATP diphosphate-lyase (cyclizing) mechanism consistent with canonical adenylylcyclases.

6. Imagine a new-to-nature adenylate cyclase enzyme molecule. Describe its structural domains and active site required to catalyze the ATP diphosphate-lyase (cyclizing) reaction, producing 3’,5’-cyclic AMP as a signaling metabolite, comparable to traditional adenylylcyclase enzymes.

7. Create an artificial adenylylcyclase protein molecule designed to act as a 3’,5’-cyclic AMP synthetase. This enzyme should carry out the adenyl cyclase reaction, converting ATP into 3’,5’-cyclic AMP by the ATP diphosphate-lyase (cyclizing) pathway.

### Prompt for heme binder:

1. Design a heme binding protein using the given data: \\n- Input PDB with protein and ligand: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb"\\n- Ligand name: "HBA"\\n- Parameters file: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/theozyme/HBA/HBA.params"\\n- CST file: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/theozyme/HBA/HBA_CYS_UPO.cst"\\n- ligand atoms that should be excluded from clashchecking because they are flexible: "O1 O2 O3 O4 C5 C10"\\n- ligand atoms that need to be more exposed and the required SASA for those atoms: "C45 C46 C47" and SASA should be 10.0\\n- amino acids should be excluded from consideration when generating protein sequences: "CM"\\n- ligand atom used for aligning the rotamers: "N1", "N2", "N3", "N4"\\nHere are some properties you should try to obtain:\\n- SASA <= 0.3\\n- RMSD <= 5\\n- LDDT >= 80\\n- Terminal residue limit < 15\\n- Radius of gyration limit for protein compactness <= 30\\n- all_cst <= 1.5\\n- CMS per atom >= 3.0\\n- CYS atom is A15

<!-- Design a heme binding protein using the given data: 
- Input PDB with protein and ligand: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb"
- Ligand name: "HBA"
- Parameters file: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/theozyme/HBA/HBA.params"
- CST file: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/theozyme/HBA/HBA_CYS_UPO.cst"
- ligand atoms that should be excluded from clashchecking because they are flexible: "O1 O2 O3 O4 C5 C10"
- ligand atoms that need to be more exposed and the required SASA for those atoms: "C45 C46 C47" and SASA should be 10.0
- amino acids should be excluded from consideration when generating protein sequences: "CM"
- ligand atom used for aligning the rotamers: "N1", "N2", "N3", "N4"
Here are some properties you should try to obtain:
- SASA <= 0.3
- RMSD <= 5
- LDDT >= 80
- Terminal residue limit < 15
- Radius of gyration limit for protein compactness <= 30
- all_cst <= 1.5
- CMS per atom >= 3.0
- CYS atom is A15

<!-- ----------------------------------------
Design a heme binding protein using the given data: \\n- Input PDB with protein and ligand: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb"\\n- Ligand name: "HBA"\\n- Parameters file: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/theozyme/HBA/HBA.params"\\n- CST file: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/theozyme/HBA/HBA_CYS_UPO.cst"\\n- ligand atoms that should be excluded from clashchecking because they are flexible: "O1 O2 O3 O4 C5 C10"\\n- ligand atoms that need to be more exposed and the required SASA for those atoms: "C45 C46 C47" and SASA should be 10.0\\n- amino acids should be excluded from consideration when generating protein sequences: "CM"\\n- ligand atom used for aligning the rotamers: "N1", "N2", "N3", "N4"\\nHere are some properties you should try to obtain:\\n- SASA <= 0.3\\n- RMSD <= 20\\n- LDDT >= 70\\n- Terminal residue limit < 15\\n- Radius of gyration limit for protein compactness <= 30\\n- all_cst <= 1.5\\n- CMS per atom >= 3.0\\n- CYS atom is A15
---------------------------------------- -->

### Gromacs prompt

Run an MD simulation system for 1pga_protein.pdb in the md_workspace. Use the default values for model, API key, API URL, mode, model and workspace.
                
