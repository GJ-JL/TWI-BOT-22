import os
import torch
from tqdm import tqdm
from transformers import pipeline
import pandas as pd
import json
import gc

# 加速 Huggingface 下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 数据加载
each_user_tweets = json.load(open("processed_data/id_tweet.json", 'r'))

# 特征提取器
feature_extract = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=1,
                           padding=True, truncation=True, max_length=50, add_special_tokens=True)


def tweets_embedding(batch_size=100000):
    print('Running batched feature extraction')
    total_users = len(each_user_tweets)
    batch_id = 0
    all_parts = []

    for start_idx in range(0, total_users, batch_size):
        end_idx = min(start_idx + batch_size, total_users)
        tweets_list = []

        print(f"\n🧩 正在处理用户 {start_idx} 到 {end_idx - 1}")
        for i in tqdm(range(start_idx, end_idx)):
            user_id = str(i)
            tweets = each_user_tweets.get(user_id, [])
            if len(tweets) == 0:
                total_each_person_tweets = torch.zeros(768)
            else:
                for j, each_tweet in enumerate(tweets):
                    if each_tweet is None:
                        total_word_tensor = torch.zeros(768)
                    else:
                        each_tweet_tensor = torch.tensor(feature_extract(each_tweet))  # [1, seq_len, 768]
                        total_word_tensor = sum(each_tweet_tensor[0]) / each_tweet_tensor.shape[1]

                    if j == 0:
                        total_each_person_tweets = total_word_tensor
                    elif j == 20:
                        break
                    else:
                        total_each_person_tweets += total_word_tensor

                total_each_person_tweets /= min(20, len(tweets))

            tweets_list.append(total_each_person_tweets)

        # 保存当前批次（可选，节省中间调试）
        part_path = f"processed_data/tweets_tensor_part_{batch_id}.pt"
        torch.save(torch.stack(tweets_list), part_path)
        print(f"✅ 已保存 {part_path}，包含 {len(tweets_list)} 个用户特征")
        all_parts.append(part_path)

        # 清理内存
        del tweets_list
        gc.collect()
        batch_id += 1

    # 合并所有批次文件并保存为最终结果
    print("\n🚀 正在合并所有批次特征...")
    all_tensors = [torch.load(p) for p in all_parts]
    full_tensor = torch.cat(all_tensors, dim=0)

    # ✅ 最终保存为指定文件名
    torch.save(full_tensor, "processed_data/tweets_tensor.pt")
    print(f"🎉 全部特征保存完成，共 {full_tensor.shape[0]} 个用户，已保存为 tweets_tensor.pt")


tweets_embedding()
