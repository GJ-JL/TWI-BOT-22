# import os
# import re
# import math
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
#
# from dataset_tool import fast_merge, get_data_dir
#
# DATASET = "Twibot-20"
# SERVER_ID = "209"
# PROCESSED_DIR = "./processed_data"
# EACH_USER_TWEETS_PATH = os.path.join(PROCESSED_DIR, "each_user_tweets.npy")
# BEHAVIOR_USER_CSV = os.path.join(PROCESSED_DIR, "labeled_user_tweet_behaviors.csv")
# BEHAVIOR_GROUP_CSV = os.path.join(PROCESSED_DIR, "labeled_group_summary.csv")
# MAX_TWEETS_PER_USER = 20
#
# os.makedirs(PROCESSED_DIR, exist_ok=True)
#
# def load_each_user_tweets(path: str, num_users: int):
#     obj = np.load(path, allow_pickle=True)
#     if isinstance(obj, np.ndarray) and obj.shape == ():
#         obj = obj.item()
#
#     if isinstance(obj, dict):
#         out = []
#         for i in range(num_users):
#             v = obj.get(i, [])
#             out.append(list(v) if v is not None else [])
#         return out
#
#     out = list(obj)
#     if len(out) < num_users:
#         out = out + [[] for _ in range(num_users - len(out))]
#     elif len(out) > num_users:
#         out = out[:num_users]
#     return out
#
# def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
#     x = x.astype(np.float64); y = y.astype(np.float64)
#     nx, ny = len(x), len(y)
#     if nx < 2 or ny < 2:
#         return float("nan")
#     vx = x.var(ddof=1); vy = y.var(ddof=1)
#     pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
#     if pooled <= 0:
#         return float("nan")
#     return float((x.mean() - y.mean()) / math.sqrt(pooled))
#
# def analyze_labeled_tweet_behaviors(user_df, tweet_text_list, each_user_tweets_list):
#     dataset_dir = get_data_dir(SERVER_ID) / DATASET
#     label_path = dataset_dir / "label.csv"
#     lab = pd.read_csv(label_path)
#
#     uid2label = dict(zip(lab["id"].astype(str), lab["label"].astype(str)))
#     user_ids = user_df["id"].astype(str).tolist()
#
#     labeled_idx = [i for i, uid in enumerate(user_ids) if uid in uid2label]
#     print(f"[check] total users={len(user_ids)}, labeled users={len(labeled_idx)}")
#     if len(labeled_idx) == 0:
#         print("No labeled users found (check label.csv path / ids).")
#         return
#
#     url_re = re.compile(r"(https?://\S+)|(t\.co/\S+)", re.IGNORECASE)
#     mention_re = re.compile(r"@\w+")
#     hashtag_re = re.compile(r"#\w+")
#     repeated_re = re.compile(r"(.)\1{2,}")
#
#     def per_tweet_features(t: str):
#         t = str(t).strip()
#         L = len(t)
#         if L == 0:
#             return None
#
#         url_cnt = len(url_re.findall(t))
#         mention_cnt = len(mention_re.findall(t))
#         hashtag_cnt = len(hashtag_re.findall(t))
#
#         is_rt = 1 if t.startswith("RT ") else 0
#         is_reply = 1 if t.startswith("@") else 0
#         has_url = 1 if url_cnt > 0 else 0
#         has_repeat = 1 if repeated_re.search(t) else 0
#
#         tokens = t.split()
#         token_cnt = len(tokens)
#         avg_token_len = (sum(len(x) for x in tokens) / token_cnt) if token_cnt > 0 else 0.0
#
#         return {
#             "len_char": L,
#             "token_cnt": token_cnt,
#             "avg_token_len": avg_token_len,
#             "url_cnt": url_cnt,
#             "mention_cnt": mention_cnt,
#             "hashtag_cnt": hashtag_cnt,
#             "is_rt": is_rt,
#             "is_reply": is_reply,
#             "has_url": has_url,
#             "has_repeat": has_repeat,
#         }
#
#     rows = []
#     for ui in tqdm(labeled_idx, desc="Extract tweet behaviors for labeled users"):
#         uid = user_ids[ui]
#         y = uid2label[uid]
#
#         idx_list = each_user_tweets_list[ui] or []
#         idx_list = list(idx_list)
#         tweet_cnt_total = len(idx_list)
#
#         feats = []
#         for j in range(min(tweet_cnt_total, MAX_TWEETS_PER_USER)):
#             ti = int(idx_list[j])
#             if 0 <= ti < len(tweet_text_list):
#                 tx = tweet_text_list[ti]
#                 if tx is None or (isinstance(tx, float) and np.isnan(tx)):
#                     continue
#                 f = per_tweet_features(tx)
#                 if f is not None:
#                     feats.append(f)
#
#         if len(feats) == 0:
#             rows.append({
#                 "user_id": uid,
#                 "label": y,
#                 "tweet_cnt_total": tweet_cnt_total,
#                 "tweet_cnt_used": 0,
#                 "len_char_mean": 0.0,
#                 "token_cnt_mean": 0.0,
#                 "avg_token_len_mean": 0.0,
#                 "rt_ratio": 0.0,
#                 "reply_ratio": 0.0,
#                 "url_ratio": 0.0,
#                 "repeat_ratio": 0.0,
#                 "url_cnt_mean": 0.0,
#                 "mention_cnt_mean": 0.0,
#                 "hashtag_cnt_mean": 0.0,
#             })
#             continue
#
#         df_f = pd.DataFrame(feats)
#         rows.append({
#             "user_id": uid,
#             "label": y,
#             "tweet_cnt_total": tweet_cnt_total,
#             "tweet_cnt_used": len(df_f),
#             "len_char_mean": float(df_f["len_char"].mean()),
#             "token_cnt_mean": float(df_f["token_cnt"].mean()),
#             "avg_token_len_mean": float(df_f["avg_token_len"].mean()),
#             "rt_ratio": float(df_f["is_rt"].mean()),
#             "reply_ratio": float(df_f["is_reply"].mean()),
#             "url_ratio": float(df_f["has_url"].mean()),
#             "repeat_ratio": float(df_f["has_repeat"].mean()),
#             "url_cnt_mean": float(df_f["url_cnt"].mean()),
#             "mention_cnt_mean": float(df_f["mention_cnt"].mean()),
#             "hashtag_cnt_mean": float(df_f["hashtag_cnt"].mean()),
#         })
#
#     df_user = pd.DataFrame(rows)
#     df_user.to_csv(BEHAVIOR_USER_CSV, index=False)
#     print(f"Saved user-level behaviors: {BEHAVIOR_USER_CSV}")
#
#     metrics = [
#         "tweet_cnt_total", "tweet_cnt_used",
#         "len_char_mean", "token_cnt_mean", "avg_token_len_mean",
#         "rt_ratio", "reply_ratio", "url_ratio", "repeat_ratio",
#         "url_cnt_mean", "mention_cnt_mean", "hashtag_cnt_mean",
#     ]
#     labels = sorted(df_user["label"].unique().tolist())
#
#     if len(labels) == 2:
#         a, b = labels[0], labels[1]
#         da = df_user[df_user["label"] == a]
#         db = df_user[df_user["label"] == b]
#         effect_rows = []
#         for m in metrics:
#             x = da[m].to_numpy()
#             y = db[m].to_numpy()
#             effect_rows.append({
#                 "metric": m,
#                 f"mean_{a}": float(np.mean(x)),
#                 f"mean_{b}": float(np.mean(y)),
#                 "mean_diff_(a-b)": float(np.mean(x) - np.mean(y)),
#                 "cohen_d_(a-b)": cohen_d(x, y),
#             })
#         pd.DataFrame(effect_rows).to_csv(BEHAVIOR_GROUP_CSV, index=False)
#     else:
#         grp = df_user.groupby("label")[metrics].agg(["count", "mean", "median"])
#         grp.columns = ["_".join(col).strip() for col in grp.columns.values]
#         grp.reset_index().to_csv(BEHAVIOR_GROUP_CSV, index=False)
#
#     print(f"Saved group summary: {BEHAVIOR_GROUP_CSV}")
#
# # ===== main =====
# user, tweet = fast_merge(dataset=DATASET, server_id=SERVER_ID)
# tweet_text = [t for t in tweet["text"]]
# each_user_tweets = load_each_user_tweets(EACH_USER_TWEETS_PATH, num_users=len(user))
#
# analyze_labeled_tweet_behaviors(user, tweet_text, each_user_tweets)
import os
import re
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset_tool import fast_merge, get_data_dir

DATASET = "Twibot-20"
SERVER_ID = "209"
PROCESSED_DIR = "./processed_data"
EACH_USER_TWEETS_PATH = os.path.join(PROCESSED_DIR, "each_user_tweets.npy")

# ===== 只在 train split 上做行为统计 =====
SPLIT_NAME = "train"

BEHAVIOR_USER_CSV = os.path.join(
    PROCESSED_DIR, f"labeled_user_tweet_behaviors_{SPLIT_NAME}.csv"
)
BEHAVIOR_GROUP_CSV = os.path.join(
    PROCESSED_DIR, f"labeled_group_summary_{SPLIT_NAME}.csv"
)

MAX_TWEETS_PER_USER = 20
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_each_user_tweets(path: str, num_users: int):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()

    if isinstance(obj, dict):
        out = []
        for i in range(num_users):
            v = obj.get(i, [])
            out.append(list(v) if v is not None else [])
        return out

    out = list(obj)
    if len(out) < num_users:
        out = out + [[] for _ in range(num_users - len(out))]
    elif len(out) > num_users:
        out = out[:num_users]
    return out

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64); y = y.astype(np.float64)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = x.var(ddof=1); vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / math.sqrt(pooled))

def analyze_labeled_tweet_behaviors(user_df, tweet_text_list, each_user_tweets_list):
    dataset_dir = get_data_dir(SERVER_ID) / DATASET
    lab = pd.read_csv(dataset_dir / "label.csv")
    uid2label = dict(zip(lab["id"].astype(str), lab["label"].astype(str)))

    # ===== 关键：只保留 train split 用户 =====
    user_df_train = user_df[user_df["split"] == SPLIT_NAME].copy()
    user_ids_train = user_df_train["id"].astype(str).tolist()
    user_indices_train = user_df_train.index.tolist()  # 对应全量 user_index

    # 只保留 train 且有 label 的用户
    labeled_train = [
        (ui, uid)
        for ui, uid in zip(user_indices_train, user_ids_train)
        if uid in uid2label
    ]

    print(f"[check] total users(train)={len(user_df_train)}, labeled users(train)={len(labeled_train)}")
    if len(labeled_train) == 0:
        print("No labeled users found in train split.")
        return

    url_re = re.compile(r"(https?://\S+)|(t\.co/\S+)", re.IGNORECASE)
    mention_re = re.compile(r"@\w+")
    hashtag_re = re.compile(r"#\w+")
    repeated_re = re.compile(r"(.)\1{2,}")

    def per_tweet_features(t: str):
        t = str(t).strip()
        L = len(t)
        if L == 0:
            return None

        url_cnt = len(url_re.findall(t))
        mention_cnt = len(mention_re.findall(t))
        hashtag_cnt = len(hashtag_re.findall(t))

        is_rt = 1 if t.startswith("RT ") else 0
        is_reply = 1 if t.startswith("@") else 0
        has_url = 1 if url_cnt > 0 else 0
        has_repeat = 1 if repeated_re.search(t) else 0

        tokens = t.split()
        token_cnt = len(tokens)
        avg_token_len = (sum(len(x) for x in tokens) / token_cnt) if token_cnt > 0 else 0.0

        return {
            "len_char": L,
            "token_cnt": token_cnt,
            "avg_token_len": avg_token_len,
            "url_cnt": url_cnt,
            "mention_cnt": mention_cnt,
            "hashtag_cnt": hashtag_cnt,
            "is_rt": is_rt,
            "is_reply": is_reply,
            "has_url": has_url,
            "has_repeat": has_repeat,
        }

    rows = []
    for ui, uid in tqdm(labeled_train, desc="Extract tweet behaviors (train split)"):
        y = uid2label[uid]

        idx_list = each_user_tweets_list[ui] or []
        idx_list = list(idx_list)
        tweet_cnt_total = len(idx_list)

        feats = []
        for j in range(min(tweet_cnt_total, MAX_TWEETS_PER_USER)):
            ti = int(idx_list[j])
            if 0 <= ti < len(tweet_text_list):
                tx = tweet_text_list[ti]
                if tx is None or (isinstance(tx, float) and np.isnan(tx)):
                    continue
                f = per_tweet_features(tx)
                if f is not None:
                    feats.append(f)

        if len(feats) == 0:
            rows.append({
                "user_id": uid,
                "label": y,
                "tweet_cnt_total": tweet_cnt_total,
                "tweet_cnt_used": 0,
                "len_char_mean": 0.0,
                "token_cnt_mean": 0.0,
                "avg_token_len_mean": 0.0,
                "rt_ratio": 0.0,
                "reply_ratio": 0.0,
                "url_ratio": 0.0,
                "repeat_ratio": 0.0,
                "url_cnt_mean": 0.0,
                "mention_cnt_mean": 0.0,
                "hashtag_cnt_mean": 0.0,
            })
            continue

        df_f = pd.DataFrame(feats)
        rows.append({
            "user_id": uid,
            "label": y,
            "tweet_cnt_total": tweet_cnt_total,
            "tweet_cnt_used": len(df_f),
            "len_char_mean": float(df_f["len_char"].mean()),
            "token_cnt_mean": float(df_f["token_cnt"].mean()),
            "avg_token_len_mean": float(df_f["avg_token_len"].mean()),
            "rt_ratio": float(df_f["is_rt"].mean()),
            "reply_ratio": float(df_f["is_reply"].mean()),
            "url_ratio": float(df_f["has_url"].mean()),
            "repeat_ratio": float(df_f["has_repeat"].mean()),
            "url_cnt_mean": float(df_f["url_cnt"].mean()),
            "mention_cnt_mean": float(df_f["mention_cnt"].mean()),
            "hashtag_cnt_mean": float(df_f["hashtag_cnt"].mean()),
        })

    df_user = pd.DataFrame(rows)
    df_user.to_csv(BEHAVIOR_USER_CSV, index=False)
    print(f"Saved user-level behaviors: {BEHAVIOR_USER_CSV}")

    metrics = [
        "tweet_cnt_total", "tweet_cnt_used",
        "len_char_mean", "token_cnt_mean", "avg_token_len_mean",
        "rt_ratio", "reply_ratio", "url_ratio", "repeat_ratio",
        "url_cnt_mean", "mention_cnt_mean", "hashtag_cnt_mean",
    ]

    labels = sorted(df_user["label"].unique().tolist())
    if len(labels) == 2:
        a, b = labels[0], labels[1]
        da = df_user[df_user["label"] == a]
        db = df_user[df_user["label"] == b]

        effect_rows = []
        for m in metrics:
            x = da[m].to_numpy()
            y = db[m].to_numpy()
            effect_rows.append({
                "metric": m,
                f"mean_{a}": float(np.mean(x)),
                f"mean_{b}": float(np.mean(y)),
                "mean_diff_(a-b)": float(np.mean(x) - np.mean(y)),
                "cohen_d_(a-b)": cohen_d(x, y),
                f"n_{a}": int(len(da)),
                f"n_{b}": int(len(db)),
            })

        pd.DataFrame(effect_rows).to_csv(BEHAVIOR_GROUP_CSV, index=False)

    print(f"Saved group summary: {BEHAVIOR_GROUP_CSV}")

# ===== main =====
user, tweet = fast_merge(dataset=DATASET, server_id=SERVER_ID)
tweet_text = [t for t in tweet["text"]]
each_user_tweets = load_each_user_tweets(EACH_USER_TWEETS_PATH, num_users=len(user))

analyze_labeled_tweet_behaviors(user, tweet_text, each_user_tweets)
