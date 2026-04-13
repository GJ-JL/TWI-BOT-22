import pandas as pd
import numpy as np
import torch
import json
import os

print('Loading raw data...')
path = './datasets/Twibot-22/'

user = pd.read_json(path + 'user.json')
edge = pd.read_csv(path + 'edge.csv')
user_idx = user['id']
uid_index = {uid: index for index, uid in enumerate(user_idx.values)}

# 创建存储目录
os.makedirs('./temp_tweet_chunks', exist_ok=True)

print("Processing tweets file-by-file...")

for i in range(9):
    name = f'tweet_{i}.json'
    filepath = os.path.join(path, name)
    print(f"  Processing {name}")

    with open(filepath, 'r') as f:
        user_tweets = json.load(f)

    # 每次处理一个文件的局部缓存
    partial_id_tweet = {}
    for each in user_tweets:
        uid = 'u' + str(each['author_id'])
        text = each['text']
        try:
            index = uid_index[uid]
            if index not in partial_id_tweet:
                partial_id_tweet[index] = []
            partial_id_tweet[index].append(text)
        except KeyError:
            continue

    # 保存本批次结果为临时 json
    with open(f'./temp_tweet_chunks/partial_{i}.json', 'w') as f:
        json.dump(partial_id_tweet, f)

print("Merging all partial results...")

# 初始化完整结构
id_tweet = {i: [] for i in range(len(user_idx))}

# 合并所有 partial 文件
for i in range(9):
    with open(f'./temp_tweet_chunks/partial_{i}.json', 'r') as f:
        partial = json.load(f)
    for k, v in partial.items():
        id_tweet[int(k)].extend(v)

# 保存最终结果
os.makedirs('./processed_data', exist_ok=True)
with open('./processed_data/id_tweet.json', 'w') as f:
    json.dump(id_tweet, f)

print("✅ All tweets processed and merged successfully.")
