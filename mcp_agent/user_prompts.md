### Prompt for enzygen:

Design an enzyme using the given data: 
- enzyme family = "4.6.1" 
- Motif sequence = "DIG"
- Coordinates for the motif sequence (X1, Y1, Z1, X2, Y2, Z2) = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0] 
- Indices of the the motif sequence = [0, 1, 4] 
- PDB file = "5cxl.A" 
- EC4 file = "4.6.1.1" 
- Substrate file = "CHEBI_57540.sdf" 
- Recommended length = 20 

<!-- ----------------------------------------
Design an enzyme using the given data: - enzyme family = \"4.6.1\" - Motif sequence = \"DIG\" - Coordinates for the motif sequence (X1, Y1, Z1, X2, Y2, Z2) = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0] - Indices of the the motif sequence = [0, 1, 4] - PDB file = \"5cxl.A\" - EC4 file = \"4.6.1.1\" - Substrate file = \"CHEBI_57540.sdf\" - Recommended length = 20
---------------------------------------- -->

Design an enzyme using the given data: 
- enzyme family = "4.6.1" 
- Information about the motif:
    - Amino acid = "D", Coordinates = [1.0, 1.0, 1.0], Index = 0
    - Amino acid = "I", Coordinates = [2.0, 2.0, 2.0], Index = 1
    - Amino acid = "G", Coordinates = [3.0, 3.0, 3.0], Index = 4
- PDB file = "5cxl.A" 
- EC4 file = "4.6.1.1" 
- Substrate file = "CHEBI_57540.sdf" 
- Recommended length = 20 

<!-- ----------------------------------------
Design an enzyme using the given data:\\n- enzyme family = \"4.6.1\" - Information about the motif:\\n    - Amino acid = \"D\", Coordinates = [1.0, 1.0, 1.0], Index = 0\\n    - Amino acid = \"I\", Coordinates = [2.0, 2.0, 2.0], Index = 1\\n    - Amino acid = \"G\", Coordinates = [3.0, 3.0, 3.0], Index = 4\\n- PDB file = \"5cxl.A\"\\n- EC4 file = \"4.6.1.1\"\\n- Substrate file = \"CHEBI_57540.sdf\"\\n- Recommended length = 20
---------------------------------------- -->

Design an enzyme using the given data: 
- enzyme family = "4.6.1" 
- Information about the motif:
    - Amino acid = "I", Coordinates = [1.0, 1.0, 1.0], Index = 1
    - Amino acid = "G", Coordinates = [2.0, 1.8, 1.5], Index = 4
    - Amino acid = "D", Coordinates = [1.1, 1.2, 2.0], Index = 0
- PDB file = "5cxl.A" 
- EC4 file = "4.6.1.1" 
- Substrate file = "CHEBI_57540.sdf"
- Recommended length = 20 

Notes:
Can remove substrate file (need during gromacs only)
Infer category from description
Extract motifs

<!-- ----------------------------------------
Design an enzyme using the given data:\\n- enzyme family = \"4.6.1\" - Information about the motif:\\n    - Amino acid = \"I\", Coordinates = [1.0, 1.0, 1.0], Index = 1\\n    - Amino acid = \"G\", Coordinates = [2.0, 1.8, 1.5], Index = 4\\n    - Amino acid = \"D\", Coordinates = [1.1, 1.2, 2.0], Index = 0\\n- PDB file = \"5cxl.A\"\\n- EC4 file = \"4.6.1.1\"\\n- Substrate file = \"CHEBI_57540.sdf\"\\n- Recommended length = 20
---------------------------------------- -->

### Prompt for heme binder:

Design a heme binding protein using the given data: 
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
- RMSD <= 20
- LDDT >= 70
- Terminal residue limit < 15
- Radius of gyration limit for protein compactness <= 30
- all_cst <= 1.5
- CMS per atom >= 3.0
- CYS atom is A15

<!-- ----------------------------------------
Design a heme binding protein using the given data: \\n- Input PDB with protein and ligand: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb"\\n- Ligand name: "HBA"\\n- Parameters file: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/theozyme/HBA/HBA.params"\\n- CST file: "/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/theozyme/HBA/HBA_CYS_UPO.cst"\\n- ligand atoms that should be excluded from clashchecking because they are flexible: "O1 O2 O3 O4 C5 C10"\\n- ligand atoms that need to be more exposed and the required SASA for those atoms: "C45 C46 C47" and SASA should be 10.0\\n- amino acids should be excluded from consideration when generating protein sequences: "CM"\\n- ligand atom used for aligning the rotamers: "N1", "N2", "N3", "N4"\\nHere are some properties you should try to obtain:\\n- SASA <= 0.3\\n- RMSD <= 20\\n- LDDT >= 70\\n- Terminal residue limit < 15\\n- Radius of gyration limit for protein compactness <= 30\\n- all_cst <= 1.5\\n- CMS per atom >= 3.0\\n- CYS atom is A15
---------------------------------------- -->
