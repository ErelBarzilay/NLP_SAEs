import os
import sys
import pandas as pd
import numpy as np
from utils import split_dataframe, get_frequency_df

def generate_df(path):
    df = pd.read_csv(path) #should be data.csv
    df = df["synonyms"]
    df = df.to_list()
    return df

def get_pre_tokens_df(path): #should be data.csv
    if (os.path.exists("pd_" + path)):
        return pd.read_csv("pd_" + path)
    df = generate_df(path)
    new_df = []
    i = 0
    num = 0
    for row in df:
        i += 1
        row = row[1:-1]
        words = row.split(",")
        for part in words:
            word = part.split(":")[0]
            word = word.strip()[1:-1]
            new_df.append([word, i, num])
            num += 1
    new_df = pd.DataFrame(new_df)
    new_df.columns = ["text", "dict_val", "enumerator"]
    new_df.to_csv("pd_" + path)
    os.chmod("pd_" + path, 0o777)
    return new_df

def get_tokens_cols(df, tokenizer):
    lst_len = []
    lst_val = []
    for _, row in df.iterrows():
        lst_val.append(tokenizer(row["prompt"])['input_ids'])
        lst_len.append(len(lst_val[len(lst_val) - 1]))
    return lst_val, lst_len

def get_freq_df(df, index):
    df = get_frequency_df(df, index)
    df["prompt"] = df["text"].apply(lambda x: f"This is a document about {x}")
    df = df[df["frequency"] > 0]
    df["log_freq"] = np.log10(df["frequency"])
    # remove 5 percednt outliers in terms of frequency
    df = df[df["frequency"] < df["frequency"].quantile(0.95)]
    return df
    
def get_token_df(path, tokenizer, index):
    df = get_freq_df(get_pre_tokens_df(path), index)
    lst_val, lst_len = get_tokens_cols(df, tokenizer)
    df["tokens"] = lst_val
    df["len"] = lst_len
    return df

def add_index_col(df):
    lst = []
    for _, row in df.iterrows():
        lst.append(df[(df["len"] < row["len"]) | ((df["len"] == row["len"]) & (df["enumerator"] < row["enumerator"]))].shape[0])
    df["enumerator"] = lst
    return df

def combine_df(data):
    data = data[data["frequency"] > 0]
    df = pd.merge(data, data, on = 'dict_val', suffixes = ('_1', '_2'))
    df = df[(df['frequency_1'] < df['frequency_2']) | ((df['frequency_1'] == df['frequency_2']) & (df['text_1'] < df['text_2']) & (df['text_1'] != df['text_2']))]
    df["diff"] = df["frequency_2"] - df["frequency_1"]
    df["frequency_1"] = df["frequency_1"].replace(0,1)
    df["frequency_2"] = df["frequency_2"].replace(0,1)
    df["log_diff"] = df["log_freq_2"] - df["log_freq_1"]
    df["loss_diff"] = df["loss_2"] - df["loss_1"]
    df["activated_features_diff"] = df["activated_features_2"] - df["activated_features_1"]
    return df
## the intended flow: 
## -- get the basic dataset (get_token_df)
## -- then get repr, post, pre
## -- you can do loss analysis as this point
## -- also can do repr and similarity analysis

    