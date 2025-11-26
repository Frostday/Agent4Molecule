import json

data_file = "/jet/home/eshen3/Agent4Molecule/mcp_agent/data/mined_motifs.json"

with open(data_file, "r") as f:
    data = json.load(f)

# Find the corresponding ec4 entry for pdb == 6eqj.A
for ec_number, entries in data.items():
    for entry in entries:
        if entry["pdb"] == "2gzl.A":
            print(f"EC Number: {ec_number}")
            # print(f"Entry: {entry}")
            break

# print(data["1.1.1.201"][0])
# 1.1.1.239
# print(data["1.1.1.239"][0])
# 1.1.1.270
# print(data["1.1.1.270"][0])
# # 1.1.1.201
# print(data["1.1.1.201"][0])
# # 1.1.1.184
# print(data["1.1.1.184"][0])
# # 1.1.1.25
# print(data["1.1.1.25"][0])
# # 1.1.1.62
# print(data["1.1.1.62"][0])
# # 1.1.1.35
# print(data["1.1.1.35"][0])
# # 1.1.1.248
# print(data["1.1.1.248"][0])
# # 1.1.1.271
# print(data["1.1.1.271"][0])
# # 1.1.1.372
# print(data["1.1.1.372"][0])
# # 1.1.1.357
# print(data["1.1.1.357"][0])
# # 1.1.1.27
# print(data["1.1.1.27"][0])
# # 1.1.1.286
# print(data["1.1.1.286"][0])
# # 1.1.1.47
# print(data["1.1.1.47"][0])
# # 1.1.1.3
# print(data["1.1.1.3"][0])

# print(data["2.8.3.28"][0])
# randomly pick 5 entries from data
# import random
# sampled_data = random.sample(list(data.keys()), 5)
# for ec in sampled_data:
#     print(f"EC Number: {ec}")
#     print(f"Sample Entry: {data[ec][0]}")
#     print()