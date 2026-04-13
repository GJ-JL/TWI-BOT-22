import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import math
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dataset_tool import fast_merge

# -------------------------
# 配置
# -------------------------
OUT_DIR = "./processed_data"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512          # description embedding 的 batch（按显存调）
MAX_TWEETS_PER_USER = 20   # 只取前 20 条推文
EMB_DTYPE = torch.float32  # 输出 dtype（如需省空间可改 float16）

# -------------------------
# 载入数据
# -------------------------
user, tweet = fast_merge(dataset="Twibot-20")

# 统一把 None/NaN 转为空字符串，避免 encode 报错
def normalize_text(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return str(x)

user_text = [normalize_text(x) for x in list(user["description"])]

# 这里建议用 tweet["text"] 更稳（你原来是 tweet.text）
tweet_text = [normalize_text(x) for x in list(tweet["text"])]

# each_user_tweets：原作者用 np.save 保存，因此应 np.load(..., allow_pickle=True)
each_user_tweets_path = os.path.join(OUT_DIR, "each_user_tweets.npy")
if not os.path.exists(each_user_tweets_path):
    each_user_tweets_path = "./processed_data/each_user_tweets.npy"

try:
    each_user_tweets = np.load(each_user_tweets_path, allow_pickle=True).item()
except Exception:
    # 兼容你之前若用 torch.save 保存的情况
    each_user_tweets = torch.load(each_user_tweets_path)

# each_user_tweets 如果是 dict：key 为 user_index；如果是 list/np.ndarray：按索引取
def get_user_tweet_ids(i):
    if isinstance(each_user_tweets, dict):
        return each_user_tweets.get(i, [])
    else:
        return each_user_tweets[i]

# -------------------------
# 模型
# -------------------------
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)

def embed_text_batch(texts, batch_size=BATCH_SIZE):
    # SentenceTransformer 内部会按 batch_size 迭代
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=False
    )

# -------------------------
# 1) 单线程：分批 encode 并拼接保存（用于 description）
# -------------------------
def generate_embeddings_concat(text_list, out_path, batch_size=BATCH_SIZE):
    """
    单线程按 batch encode，最后 torch.cat 后保存为一个大 tensor。
    """
    all_emb = []
    total = len(text_list)

    for start in tqdm(range(0, total, batch_size), desc=f"Embedding -> {os.path.basename(out_path)}"):
        end = min(start + batch_size, total)
        batch_texts = text_list[start:end]
        emb = embed_text_batch(batch_texts, batch_size=batch_size)
        emb = emb.detach().to("cpu", dtype=EMB_DTYPE).contiguous()
        all_emb.append(emb)

    merged = torch.cat(all_emb, dim=0).contiguous()
    torch.save(merged, out_path)
    print(f"Saved embeddings to {out_path}, shape={tuple(merged.shape)}")
    return merged

# -------------------------
# 2) 推文嵌入：只取前20条 + 平均池化（每用户一个768）
# -------------------------
def generate_user_tweet_mean_embeddings(out_path):
    """
    对每个用户：
      - 取 each_user_tweets[i] 的前 MAX_TWEETS_PER_USER 条 tweet_index
      - 每条推文用 MPNet 得到 768
      - 平均池化得到用户级 768
    输出：shape [num_users, 768]
    """
    num_users = len(user)
    emb_dim = model.get_sentence_embedding_dimension()

    user_emb_list = []
    # 为了效率：把每个用户的推文集合 flatten 成批次编码（但仍然是单线程）
    # 实现：分用户 batch，避免一次 flatten 太大占内存
    USERS_PER_BLOCK = 2000  # 每块处理多少用户，可按内存调整

    for u_start in tqdm(range(0, num_users, USERS_PER_BLOCK), desc="User tweet mean embedding"):
        u_end = min(u_start + USERS_PER_BLOCK, num_users)

        # 1) 收集本块所有用户的推文文本（最多 20/用户），并记录 spans
        flat_texts = []
        spans = []  # (offset, count)
        for i in range(u_start, u_end):
            tids = get_user_tweet_ids(i)
            if tids is None or len(tids) == 0:
                spans.append((len(flat_texts), 0))
                continue

            tids = tids[:MAX_TWEETS_PER_USER]
            texts_i = []
            for tid in tids:
                try:
                    texts_i.append(tweet_text[int(tid)])
                except Exception:
                    continue

            spans.append((len(flat_texts), len(texts_i)))
            flat_texts.extend(texts_i)

        # 2) 编码本块所有推文（单线程）
        if len(flat_texts) > 0:
            flat_emb = embed_text_batch(flat_texts, batch_size=BATCH_SIZE)
            flat_emb = flat_emb.detach().to("cpu", dtype=EMB_DTYPE).contiguous()
        else:
            flat_emb = torch.empty((0, emb_dim), dtype=EMB_DTYPE)

        # 3) 回到用户级：对每个用户 spans 做 mean pooling
        block_user_emb = torch.zeros((u_end - u_start, emb_dim), dtype=EMB_DTYPE)
        for idx, (off, cnt) in enumerate(spans):
            if cnt == 0:
                continue
            block_user_emb[idx] = flat_emb[off:off + cnt].mean(dim=0)

        user_emb_list.append(block_user_emb)

    merged = torch.cat(user_emb_list, dim=0).contiguous()
    torch.save(merged, out_path)
    print(f"Saved user tweet mean embeddings to {out_path}, shape={tuple(merged.shape)}")
    return merged

# -------------------------
# 3) 行为特征：只取前20条 + z-score 标准化 + 保存为 .pt
# -------------------------
def zscore_np(x: np.ndarray) -> np.ndarray:
    """
    按原作者风格：z-score = (x-mean)/std
    防止 std=0：用 1e-8 做保护
    """
    x = x.astype(np.float32)
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-8:
        sigma = 1e-8
    return (x - mu) / sigma

def compute_behavior_features_top20():
    """
    对每个用户，仅取前20条推文计算：
      - rt_ratio
      - mention_cnt_mean
      - hashtag_cnt_mean
    """
    print("Computing behavior features (top20)...")
    num_users = len(user)

    rt_ratio = np.zeros(num_users, dtype=np.float32)
    mention_cnt_mean = np.zeros(num_users, dtype=np.float32)
    hashtag_cnt_mean = np.zeros(num_users, dtype=np.float32)

    for i in tqdm(range(num_users), desc="Behavior(top20)"):
        tids = get_user_tweet_ids(i)
        if tids is None or len(tids) == 0:
            continue

        tids = tids[:MAX_TWEETS_PER_USER]
        rt = 0
        mention = 0
        hashtag = 0
        n = 0

        for tid in tids:
            try:
                t = tweet_text[int(tid)]
            except Exception:
                continue
            n += 1
            if t.startswith("RT "):
                rt += 1
            mention += t.count("@")
            hashtag += t.count("#")

        if n == 0:
            continue

        rt_ratio[i] = rt / n
        mention_cnt_mean[i] = mention / n
        hashtag_cnt_mean[i] = hashtag / n

    return rt_ratio, mention_cnt_mean, hashtag_cnt_mean

def save_behavior_features_pt():
    rt_ratio, mention_cnt_mean, hashtag_cnt_mean = compute_behavior_features_top20()

    # z-score 标准化（与 num_properties 的处理方式一致）
    rt_ratio_z = zscore_np(rt_ratio)
    mention_z = zscore_np(mention_cnt_mean)
    hashtag_z = zscore_np(hashtag_cnt_mean)

    behavior = np.vstack([rt_ratio_z, mention_z, hashtag_z]).T  # [num_users, 3]
    behavior_tensor = torch.tensor(behavior, dtype=torch.float32)

    out_path = os.path.join(OUT_DIR, "behavior_properties_tensor.pt")
    torch.save(behavior_tensor, out_path)
    print(f"Saved behavior properties to {out_path}, shape={tuple(behavior_tensor.shape)}")
    return behavior_tensor

# -------------------------
# 主流程
# -------------------------
def main():
    print("Processing behavior_features...")
    # 3) 行为特征（z-score 后 .pt）
    save_behavior_features_pt()

    # 1) description embeddings：单线程分批 encode + cat
    print("Processing description embeddings...")
    generate_embeddings_concat(user_text, os.path.join(OUT_DIR, "des_tensor_enh.pt"), batch_size=BATCH_SIZE)

    # 2) tweet embeddings：只取前20条 + mean pooling（每用户一个768）
    print("Processing tweet embeddings (top20 mean pooled)...")
    generate_user_tweet_mean_embeddings(os.path.join(OUT_DIR, "tweets_tensor_enh.pt"))

if __name__ == "__main__":
    main()
