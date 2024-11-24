import numpy as np
import token_counter
from tqdm import tqdm
import pandas as pd 

def split_dataframe(df, chunk_size = 1000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:min((i+1)*chunk_size, df.shape[0])])
    return chunks

def different(lst1, lst2):
    if len(lst1) != len(lst2):
        return True
    else:
        for i in range(len(lst1)):
            if lst1[i] != lst2[i]:
                return True
    return False

def cosine_similarity(x_1, x_2):
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    return np.dot(x_1, x_2)/np.sqrt((np.dot(x_1, x_1) * np.dot(x_2, x_2)))

def jaccard_similarity(x_1, x_2):
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    intersection = np.sum(np.logical_and(x_1 != 0, x_2 != 0))
    union = np.sum(np.logical_or(x_1 != 0, x_2 != 0))
    return intersection/union

def get_frequency_df(df, index = 'v4_pileval_gpt2'):
    df["frequency"] = token_counter.count_tokens_dataset(df, 60, index = index)
    return df

def different(lst1, lst2):
    if len(lst1) != len(lst2):
        return True
    else:
        for i in range(len(lst1)):
            if lst1[i] != lst2[i]:
                return True
    return False

def add_row_to_df(df, f, name, vec):
    add_col = []
    for _, row in tqdm(df.iterrows(), total = df.shape[0]):
        add_col.append(f(vec[row["enumerator_1"]], vec[row["enumerator_2"]]))
    df[name] = add_col
    return df
    
if __name__ == '__main__':
    print()