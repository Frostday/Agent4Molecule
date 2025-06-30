import os
import sys
sys.path.append("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/")
from typing import Annotated
from pydantic import Field
import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("enzygen")

@mcp.tool()
def get_motif_sequence(enzyme_family: Annotated[str, Field(description="Enzyme family name")],
                       motif_seq: Annotated[str, Field(description="Sequence of the motif")],
                       motif_coord: Annotated[list[int], Field(description="Coordinates of the motif")],
                       motif_indices: Annotated[list[int], Field(description="Indices of the motif")],
                       motif_pdb: Annotated[str, Field(description="PDB file of the motif")],
                       motif_ec4: Annotated[str, Field(description="EC4 file of the motif")],
                       motif_substrate: Annotated[str, Field(description="Substrate file of the motif")],
                       recommended_length: Annotated[int, Field(description="Recommended length of the motif")]):
    file_name = "/ocean/projects/cis240137p/dgarg2/github/EnzyGen/data/input.json"
    data = {}
    indices, pdb, ec4, substrate = ",".join([str(i) for i in motif_indices])+"\n", motif_pdb, motif_ec4, motif_substrate
    seq, coord = "", ""
    idx = 0
    for i in range(recommended_length):
        if i in motif_indices:
            seq += motif_seq[idx]
            coord += ",".join([str(i) for i in motif_coord[idx*3:idx*3+3]])+","
            idx += 1
        else:
            seq += "A"
            coord += "0.0,0.0,0.0,"
    coord = coord[:-1]
    data = {
        enzyme_family: {
            "test": {
                "seq": [seq],
                "coor": [coord],
                "motif": [indices],
                "pdb": [pdb],
                "ec4": [ec4],
                "substrate": [substrate]
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
    text = f"""#!/bin/bash\n\nrm -rf outputs/*\n\ndata_path={input_json}\n\noutput_path=models\nproteins=({" ".join(enzymes_families)})\n\nfor element in ${{proteins[@]}}\ndo\ngeneration_path=/ocean/projects/cis240137p/dgarg2/github/EnzyGen/outputs/${{element}}\n\nmkdir -p ${{generation_path}}\nmkdir -p ${{generation_path}}/pred_pdbs\nmkdir -p ${{generation_path}}/tgt_pdbs\n\n/ocean/projects/cis240137p/dgarg2/miniconda3/envs/enzygen/bin/python fairseq_cli/validate.py ${{data_path}} --task geometric_protein_design --protein-task ${{element}} --dataset-impl-source "raw" --dataset-impl-target "coor" --path ${{output_path}}/checkpoint_best.pt --batch-size 1 --results-path ${{generation_path}} --skip-invalid-size-inputs-valid-test --valid-subset test --eval-aa-recovery\ndone"""
    run_file = "/ocean/projects/cis240137p/dgarg2/github/EnzyGen/run_enzygen.sh"
    with open(run_file, "w") as f:
        f.write(text)
    os.system(f"chmod +x {run_file}")
    os.system(f"cd /ocean/projects/cis240137p/dgarg2/github/EnzyGen && bash {run_file}")
    output_preds = []
    for enzyme_family in os.listdir("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/outputs"):
        for output in os.listdir(f"/ocean/projects/cis240137p/dgarg2/github/EnzyGen/outputs/{enzyme_family}/pred_pdbs"):
            if output.endswith(".pdb"):
                with open(f"/ocean/projects/cis240137p/dgarg2/github/EnzyGen/outputs/{enzyme_family}/pred_pdbs/{output}", "r") as f:
                    content = f.read()
                output_preds.append(enzyme_family + "\n" + output + "\n\n" + content)
    return "Predicted structure(s) from EnzyGen:\n\n" + "\n----------\n".join(output_preds)


if __name__ == "__main__":
    mcp.run(transport='stdio')
    # print(get_motif_sequence("4.6.1", "DIG", [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], [0, 1, 4], "5cxl.A", "4.6.1.1", "CHEBI_57540.sdf", 5))
    # print(run_enzygen("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/data/input.json"))
    # print(run_enzygen("/ocean/projects/cis240137p/dgarg2/github/EnzyGen/data/test_2.json"))
