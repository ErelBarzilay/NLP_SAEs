import requests
import numpy as np
from tqdm import tqdm
from threading import Thread
import os
import json
def count_tokens(text, lst, num, index):
    payload = {
        'index': index,
        'query_type': 'count',
        'query': text,
    }
    result = requests.post('https://api.infini-gram.io/', json=payload).json()
    lst[num] = result["count"]


def count_tokens_dataset(df, num_threads, index):
    if os.path.exists(f"data_save/token_count_{index}.npy"):
        lst = np.load(f"data_save/token_count_{index}.npy")
        return np.array(lst)
    else:
        lst = [None for _ in range(len(df["text"]))]
        threads = [None for _ in range(num_threads)]
        num = 0
        for entry in tqdm(df["text"]):
            threads[num % num_threads] = Thread(target = count_tokens, args = (entry, lst, num, index))
            threads[num % num_threads].start()
            if num % num_threads == 0 and num > 0:
                for i in range(num_threads):
                    threads[i].join()
            num += 1
            if num == len(df["text"]):
                for i in range(num % num_threads):
                    threads[i].join()
        lst = np.array(lst)
        np.save(f"data_save/token_count_{index}.npy", lst)
        os.chmod(f"data_save/token_count_{index}.npy", 0o777)
        return lst

if __name__ == '__main__':
    text = 'Erel OR Erel OR Erel OR Erel'
    print(count_tokens(text))
    if count_tokens(text) > 0:
        print("(°ｏ°)")