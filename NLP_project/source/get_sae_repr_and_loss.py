import sys 
import torch
import os
import dataset_generator
from tqdm import tqdm
sys.path.insert(0,"/home/joberant/NLP_2324b/erelbarzilay/conda_3/envs/new_env/lib/python3.12/site-packages")
from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
import pandas as pd
import numpy as np
from utils import split_dataframe, get_frequency_df, jaccard_similarity, cosine_similarity, add_row_to_df
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3
sys.path.insert(0,"/vol/joberant_nobck/data/NLP_368307701_2324b/erelbarzilay/conda_3/envs/new_env/lib/python3.12/site-packages")
from sae_lens import SAE
from omegaconf import OmegaConf, ListConfig
import argparse
import datetime
from itertools import product
import traceback


# if torch.backends.mps.is_available():
#     device = "mps"
# else:
#     device = "cuda" if torch.cuda.is_available() else "cpu"
### delete this when edo saves me
# device = "cuda"
### delete this when edo saves me
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

## why doesnt this work?
login(TOKEN)
print("device: ", device)
max_token_len = 200
def split_by_len(df):
    lst = []
    for i in range(1, max_token_len):
        lst.append([])
        df_lst = split_dataframe(df[df["len"] == i])
        for data in df_lst:
            lst[i - 1].append(Dataset.from_pandas(data))
    return lst

def test(lst, data):
    new_lst = []
    for num, l in enumerate(lst):
        new_lst.append([])
        for v in l:
            new_lst[num].append(Dataset.to_pandas(v))
        new_lst[num] = pd.concat(new_lst[num])
    total_so_far = 0
    for i in range(1, max_token_len):
        curr = 0
        for _, row in data[data["len"] == i].iterrows():
            assert row["len"] == i
            assert row["tokens"] == list(new_lst[i - 1].iloc[curr]["tokens"])
            assert row["enumerator"] == curr + total_so_far
            curr += 1
        total_so_far = total_so_far + new_lst[i - 1].shape[0]
    print("test passed")
    
def get_repr_val_out(model_name, release, sae_id, index, path):
    ## model_name: the name of the LM
    ## release: the name of the sae-type for the LM (sae_lens/pretrained_saes.yaml)
    ## sae_id: the id of the sae
    ## path: the path for the data (data.csv)
    ## index: the name of the dataset we should use (check out infini-gram)
    
    for i in range(max_token_len):
        try:
            model = HookedTransformer.from_pretrained(model_name , device = device)
            print(device)
            tokenizer = model.tokenizer
            sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = release, # see options in sae_lens/pretrained_saes.yaml
            sae_id = sae_id, # won't always be a hook point
            device = device)
            break
        except:
            
            continue
    data = dataset_generator.get_token_df(path, tokenizer, index)
    data = dataset_generator.add_index_col(data)
    lst = split_by_len(data)
    test(lst, data)
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    vals = []
    features = []
    sae_outs = []
    for i in tqdm(range(1, max_token_len)):
        vals.append([])
        features.append([])
        sae_outs.append([])
        for df in lst[i - 1]:
            with torch.no_grad():
                # activation store can give us tokens.
                if len(df["prompt"]) == 0:
                    print("skipping empty df")
                    continue
                batch_tokens = model.to_tokens(df["prompt"])
                _, cache = model.run_with_cache(batch_tokens, stop_at_layer=sae.cfg.hook_layer + 1, prepend_bos=True, names_filter=[sae.cfg.hook_name])
                # Use the SAE
                val = cache[sae.cfg.hook_name]
                vals[i - 1].append(val[:,val.shape[1] - 1,:])
                feature_acts = sae.encode(cache[sae.cfg.hook_name])
                features[i - 1].append(feature_acts[:,feature_acts.shape[1] - 1,:])
                sae_out = sae.decode(feature_acts)
                sae_outs[i - 1].append(sae_out[:,sae_out.shape[1] - 1,:])

                # save some room
                del cache
    val_lst = []
    rep_lst = []
    out_lst = []
    i = 0
    for lst_vals, lst_outs, reprs in zip(vals, sae_outs, features):
        if len(lst_vals) != 0:
            val_lst.append(torch.cat(lst_vals))
            out_lst.append(torch.cat(lst_outs))
            rep_lst.append(torch.cat(reprs))
    rep_lst = torch.cat(rep_lst).detach().cpu().numpy()
    outs = torch.cat(out_lst).detach().cpu().numpy()
    vals_pre_sae = torch.cat(val_lst).detach().cpu().numpy()
    return data ,rep_lst, outs, vals_pre_sae
    
def loss(x, x_hat):return np.sqrt(np.sum(np.square(x_hat - x), axis = -1))

def add_loss_col(df, outs, vals_pre_sae, rep):
    L2_loss = []
    activated_features = []
    for _ ,row in df.iterrows():
        L2_loss.append(loss(outs[row["enumerator"]], vals_pre_sae[row["enumerator"]]))
        activated_features.append(np.count_nonzero(rep[row["enumerator"]]))
        
    df["loss"] = L2_loss
    df["activated_features"] = activated_features
    return df
    # the dataset here is pre-merge
        
def add_similarities(df, outs, vals_pre_sae, repr):
    df = add_row_to_df(df, jaccard_similarity, "Jaccard_Similarity", repr)
    df = add_row_to_df(df, cosine_similarity, "Repr_Cosine_Similarity", repr)
    df = add_row_to_df(df, cosine_similarity, "Pre_Cosine_Similarity", vals_pre_sae)
    df = add_row_to_df(df, cosine_similarity, "Post_Cosine_Similarity", outs)
    return df
    
def gen_graph(df, name, x_axis, y_axis, output_dir):
    new_df = df
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.scatter(new_df[x_axis], new_df[y_axis], alpha = 0.2, s=0.5)
    #plt.xscale("log")
    b, a = np.polyfit(new_df[x_axis], new_df[y_axis], deg = 1)
    xseq = np.linspace(0, max(new_df[x_axis]), num=100)
    ax.plot(xseq, a + b * xseq, "r--", color = "red", lw = 2.5 )
    print(b , a)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    # with open(os.path.join(output_dir, name + ".html"), "w") as f:
    #     f.write(mpld3.fig_to_html(fig))
    # os.chmod(os.path.join(os.path.join(output_dir, name + ".html")), 0o777)
    fig.savefig(os.path.join(output_dir, name + ".png"))
    os.chmod(os.path.join(output_dir, name + ".png"), 0o777)

def save_everything(data, combined_df, rep, post, pre, run_data_dir):
    data.to_csv(os.path.join(run_data_dir, "data.csv"))
    os.chmod(os.path.join(run_data_dir, "data.csv"), 0o777)
    combined_df.to_csv(os.path.join(run_data_dir, "combined_data.csv"))
    os.chmod(os.path.join(run_data_dir, "combined_data.csv"), 0o777)
    # we should also save data, rep, post, pre
    # tensors_dir = os.path.join(run_data_dir, "tensors")
    # os.mkdir(tensors_dir)
    # os.chmod(tensors_dir, 0o777)
    # torch.save(rep, os.path.join(tensors_dir, "rep.pt"))
    # torch.save(post, os.path.join(tensors_dir, "post.pt"))
    # torch.save(pre, os.path.join(tensors_dir, "pre.pt"))
    # os.chmod(os.path.join(tensors_dir, "rep.pt"), 0o777)
    # os.chmod(os.path.join(tensors_dir, "post.pt"), 0o777)
    # os.chmod(os.path.join(tensors_dir, "pre.pt"), 0o777)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", type = str)
    parser.add_argument("--sae_config", type = str, default="configs/saes_config.yaml")
    parser.add_argument("--pretrained_saes", type = str, default="configs/pretrained_saes.yaml")
    args = parser.parse_args()

    run_config = OmegaConf.load(args.run_config)
    pretrained_saes = OmegaConf.load(args.pretrained_saes)
    saes_config = OmegaConf.load(args.sae_config)

    releases = run_config.releases

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # id is the highest number in the output directory +1
    id = 0
    for f in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, f)):
            try:
                id = max(id, int(f.split(".")[0]))
            except:
                pass
    id += 1

    sample_name = f"{id}.{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    sample_dir = os.path.join(output_dir, sample_name)
    os.mkdir(sample_dir)
    os.chmod(sample_dir, 0o777)
    sample_data_dir = os.path.join(sample_dir, "data")
    os.mkdir(sample_data_dir)
    os.chmod(sample_data_dir, 0o777)
    sample_graph_dir = os.path.join(sample_dir, "graphs")
    os.mkdir(sample_graph_dir)
    os.chmod(sample_graph_dir, 0o777)

    with open(os.path.join(sample_dir, "run_config.yaml"), "w") as f:
        OmegaConf.save(run_config, f)
    with open(os.path.join(sample_dir, "pretrained_saes.yaml"), "w") as f:
        OmegaConf.save(pretrained_saes, f)
    with open(os.path.join(sample_dir, "saes_config.yaml"), "w") as f:
        OmegaConf.save(saes_config, f)

    for release in releases:

        indices = saes_config[release].index
        if not isinstance(indices, ListConfig):
            indices = [indices]

        paths = saes_config[release].path
        if not isinstance(paths, ListConfig):
            paths = [paths]

        model_names = [pretrained_saes[release].model]

        sae_ids = [sae.id for sae in pretrained_saes[release].saes]


        release_graph_dir = os.path.join(sample_graph_dir, release)
        os.makedirs(release_graph_dir, exist_ok=True)    
        release_data_dir = os.path.join(sample_data_dir, release)
        os.makedirs(release_data_dir, exist_ok=True)

        for i, (path, sae_id, model_name, index) in enumerate(product(paths, sae_ids, model_names, indices)):
            print("-------- New Model ----------")
            print (path)
            print (sae_id)
            print(model_name)
            print(index)
            
            try:
                generic_name = index + "_" + sae_id + "_" + release # model_name is determined by release, so no need to use it
                run_name = f"{i}.{generic_name}"
                run_name = run_name.replace("/", "_")
                
                run_graph_dir = os.path.join(release_graph_dir, run_name)
                os.makedirs(run_graph_dir, exist_ok=True)
                os.chmod(run_graph_dir, 0o777)
                run_data_dir = os.path.join(release_data_dir, run_name)
                os.makedirs(run_data_dir, exist_ok=True)
                os.chmod(run_data_dir, 0o777)

                data, rep, post, pre = get_repr_val_out(model_name, release, sae_id, index, path)
                data = add_loss_col(data, post, pre, rep)
                combined_df = dataset_generator.combine_df(data)
                combined_df = add_similarities(combined_df, post, pre, rep)

                save_everything(data, combined_df, rep, post, pre, run_data_dir)

                gen_graph(data, "loss", "frequency", "loss", run_graph_dir)
                gen_graph(data, "loss", "log_freq", "loss", run_graph_dir)
                gen_graph(data, "activated_features", "frequency", "activated_features", run_graph_dir)
                gen_graph(data, "activated_features", "log_freq", "activated_features", run_graph_dir)
                for string in ["loss_diff","Jaccard_Similarity", "Repr_Cosine_Similarity", "Pre_Cosine_Similarity", "Post_Cosine_Similarity", "activated_features_diff"]:
                    for by in ["log_diff", "diff"]:
                        gen_graph(combined_df, string + "_" + by, by, string, run_graph_dir)
            except Exception as e:
                print(f"Failed on {run_name}")
                print(e)
                print(traceback.format_exc())
                continue


if __name__ == '__main__':
    main()
    
# 1. Freq vs loss
# 2. Cosine (Jaccard) similarity of repr. of words given difference in frequency
# 3. Cosine similarity of value before and after sae (given difference in frequency)
