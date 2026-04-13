from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os
import pandas as pd
import json

# 加载数据
user = pd.read_json('./datasets/Twibot-22/user.json')
user_text = list(user['description'])
each_user_tweets = json.load(open("processed_data/id_tweet.json", 'r'))

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base').cuda(1)
model.eval()

# 特征提取函数
def extract_features(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        outputs = model(**{k: v.cuda(1) for k, v in inputs.items()})
    return outputs.last_hidden_state.mean(dim=1).cpu()

def Des_embbeding():
    print('Running feature1 embedding')
    path = "./processed_data/des_tensor.pt"
    if not os.path.exists(path):
        des_vec = []
        batch_size = 64  # 设置批次大小
        for start_idx in tqdm(range(0, len(user_text), batch_size)):
            end_idx = min(start_idx + batch_size, len(user_text))
            batch_texts = user_text[start_idx:end_idx]

            # 提取特征
            batch_features = extract_features(batch_texts)
            des_vec.append(batch_features)

        des_tensor = torch.cat(des_vec, 0)
        torch.save(des_tensor, path)
    else:
        des_tensor = torch.load(path)
    print('Finished')
    return des_tensor


def tweets_embedding():
    print('Running feature2 embedding')
    path = "./processed_data/tweets_tensor.pt"
    if not os.path.exists(path):
        tweets_list = []
        batch_size = 64
        for start_idx in tqdm(range(0, len(each_user_tweets), batch_size)):
            end_idx = min(start_idx + batch_size, len(each_user_tweets))
            batch_tweets = [each_user_tweets[str(i)] for i in range(start_idx, end_idx)]

            # 提取推文特征
            batch_features = []
            for user_tweets in batch_tweets:
                if len(user_tweets) == 0:
                    batch_features.append(torch.zeros(768))
                else:
                    tweet_features = []
                    for tweet in user_tweets[:20]:  # 限制推文数量
                        if tweet is None:
                            tweet_features.append(torch.zeros(768))
                        else:
                            tweet_tensor = extract_features([tweet])
                            # 确保输出的维度是 [768]，如果是 [1, 768]，则需要 squeeze
                            tweet_tensor = tweet_tensor.squeeze(0)  # 去除多余的维度
                            tweet_features.append(tweet_tensor)

                    # 对每个用户的推文特征进行平均
                    user_feature = torch.mean(torch.stack(tweet_features), dim=0)
                    batch_features.append(user_feature)

            # 确保每个批次的特征维度一致
            print(f"Batch feature size: {batch_features[0].shape}")
            tweets_list.append(torch.stack(batch_features))

        # 拼接所有的用户特征
        tweet_tensor = torch.cat(tweets_list, 0)
        torch.save(tweet_tensor, path)
    else:
        tweet_tensor = torch.load(path)
    print('Finished')


Des_embbeding()
tweets_embedding()
