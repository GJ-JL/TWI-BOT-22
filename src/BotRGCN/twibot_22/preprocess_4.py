import torch
from tqdm import tqdm
from transformers import pipeline
import os
import pandas as pd
import json
import gc

user = pd.read_json('./datasets/Twibot-22/user.json')

user_text = list(user['description'])
each_user_tweets = json.load(open("processed_data/id_tweet.json", 'r'))

feature_extract = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=1, padding=True, truncation=True, max_length=50, add_special_tokens=True)

# 断点恢复路径
DES_EMBEDDING_PATH = "./processed_data/des_tensor.pt"
TWEETS_EMBEDDING_PATH = "./processed_data/tweets_tensor.pt"
PARTIAL_TWEETS_PATH = "./processed_data/partial_tweets_tensor.pt"


def Des_embbeding():
    print('Running feature1 embedding')
    if not os.path.exists(DES_EMBEDDING_PATH):
        des_vec = []
        for k, each in enumerate(tqdm(user_text)):
            if each is None:
                des_vec.append(torch.zeros(768))
            else:
                feature = torch.Tensor(feature_extract(each))
                feature_tensor = feature[0].sum(dim=0) / feature.shape[1]
                des_vec.append(feature_tensor)

        des_tensor = torch.stack(des_vec, 0)
        torch.save(des_tensor, DES_EMBEDDING_PATH)
    else:
        des_tensor = torch.load(DES_EMBEDDING_PATH)
    print('Finished')
    return des_tensor


def tweets_embedding(batch_size=32):
    print('Running feature2 embedding')

    tweets_list = []
    start_index = 0
    total_users = len(each_user_tweets)

    # 尝试加载已有的中间结果
    if os.path.exists(PARTIAL_TWEETS_PATH):
        print("Found partial results. Loading...")
        partial_data = torch.load(PARTIAL_TWEETS_PATH)
        tweets_list = partial_data["tensor"]
        start_index = partial_data["index"] + 1
        print(f"Resuming from user {start_index}/{total_users}")

    # 从未完成的用户继续处理
    for i in tqdm(range(start_index, total_users)):
        user_tweets = each_user_tweets[str(i)]
        total_each_person_tweets = torch.zeros(768)

        if len(user_tweets) == 0:
            total_each_person_tweets = torch.zeros(768)
        else:
            for j in range(len(user_tweets)):
                each_tweet = user_tweets[j]
                if each_tweet is None:
                    total_word_tensor = torch.zeros(768)
                else:
                    each_tweet_tensor = torch.tensor(feature_extract(each_tweet))
                    total_word_tensor = each_tweet_tensor[0].sum(dim=0) / each_tweet_tensor.shape[1]
                if j == 0:
                    total_each_person_tweets = total_word_tensor
                elif j == 20:
                    break
                else:
                    total_each_person_tweets += total_word_tensor

            if j == 20:
                total_each_person_tweets /= 20
            else:
                total_each_person_tweets /= len(user_tweets)

        tweets_list.append(total_each_person_tweets)

        # 每处理完一个用户就保存
        torch.save({
            "tensor": tweets_list,
            "index": i
        }, PARTIAL_TWEETS_PATH)

        # 清理缓存
        del user_tweets, total_each_person_tweets, each_tweet_tensor, total_word_tensor
        gc.collect()
        torch.cuda.empty_cache()

    # 保存最终完整数据
    tweet_tensor = torch.stack(tweets_list)
    torch.save(tweet_tensor, TWEETS_EMBEDDING_PATH)
    print('Finished')

Des_embbeding()
tweets_embedding()
