import json

data_file = "/jet/home/eshen3/Agent4Molecule/mcp_agent/data/mined_motifs.json"

with open(data_file, "r") as f:
    data = json.load(f)

# Find the corresponding ec4 entry for pdb == 6eqj.A
for ec_number, entries in data.items():
    for entry in entries:
        if entry["pdb"] == "2gzl.A":
            print(f"EC Number: {ec_number}")
            break

