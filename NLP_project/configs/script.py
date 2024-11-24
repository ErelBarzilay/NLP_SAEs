""" First Part - Create saes_indices.yaml """
releases = []
with open("pretrained_saes.yaml", "r") as f:
    for line in f.readlines():
        if (line[0] != " " and line[0] != "\n"):
            print(line[:-2])
            # releases.append(i[:-2])
            releases.append(line)

print(releases)
output = ""

gpt2_path = ["data.csv"]
llama_path = ["data.csv"]
pythia_path = ["data.csv"]
gemma_path = ["data.csv"]
mistral_path = ["data.csv"]

# gpt2_indices = ["v4_pileval_gpt2"]
# llama_indices = ["v4_dolma-v1_7_llama"]
# pythia_indeces = ["v4_dolma-v1_7_llama"] # Can be changed
# gemma_indeces = ["v4_dolma-v1_7_llama", "v4_pileval_gpt2"] # Can be changed
# mistral_indeces = ["v4_dolma-v1_7_llama"] # Can be changed

gpt2_indices = ["v4_pileval_gpt2"]
llama_indices = ["v4_pileval_llama"]
pythia_indeces = ["v4_pileval_llama", "v4_pileval_gpt2"] # Can be changed
gemma_indeces = ["v4_pileval_llama", "v4_pileval_gpt2"] # Can be changed
mistral_indeces = ["v4_pileval_llama", "v4_pileval_gpt2"] # Can be changed

paths  =  {
                "gpt2": gpt2_path,
                "llama": llama_path, 
                "pythia":pythia_path, 
                "gemma": gemma_path, 
                "mistral":mistral_path
            }

indices  =  {
                "gpt2": gpt2_indices,
                "llama": llama_indices, 
                "pythia":pythia_indeces, 
                "gemma": gemma_indeces, 
                "mistral":mistral_indeces
            }
for release in releases:
    output += release
    model = ""

    if ("gpt2" in release):
        model = "gpt2"
    if ("llama" in release):
        model = "llama"
    if ("pythia" in release):
        model = "pythia"
    if ("gemma" in release):
        model = "gemma"
    if ("mistral" in release):
        model = "mistral"

    output += "  path:\n"
    for path in paths[model]:
        output += "    - " + path + "\n"

    output += "  index:\n"
    for index in indices[model]:
        output += "    - " + index + "\n"
        

with open("saes_config.yaml", "w") as f:
    f.write(output)

""" Second Part - Create run_config.yaml """

output = "releases:\n"

for release in releases:
    # output += release

    output += "  - " + release[:-2] + "\n"

with open("run_config.yaml", "w") as f:
    f.write(output)