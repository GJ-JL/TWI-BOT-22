import os
import re
import json
import argparse
import torch
from tqdm import tqdm
from dataset_tool import fast_merge
from sentence_transformers import SentenceTransformer

# =========================
# Cleaning / normalization
# =========================
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")

def clean_tweet(text: str, max_chars: int = 220) -> str:
    """Basic tweet cleaning for stable representation."""
    if text is None:
        return ""
    t = text.strip()
    t = URL_RE.sub("<URL>", t)
    t = MENTION_RE.sub("<USER>", t)
    t = re.sub(r"\s+", " ", t).strip()
    if max_chars > 0 and len(t) > max_chars:
        t = t[:max_chars].rstrip() + "…"
    return t

def normalize_for_dup(text: str) -> str:
    """Normalization used for duplicate/template estimation."""
    t = (text or "").strip().lower()
    t = URL_RE.sub("<URL>", t)
    t = MENTION_RE.sub("<USER>", t)
    t = NUM_RE.sub("<NUM>", t)
    # keep hashtags and <...> tags, drop other punctuation
    t = re.sub(r"[^\w\s<>#]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def lexical_diversity(cleaned_texts):
    tokens = []
    for t in cleaned_texts:
        tokens.extend(re.findall(r"[A-Za-z]+", t.lower()))
    if not tokens:
        return 0.0
    return len(set(tokens)) / float(len(tokens))

# =========================
# Structured doc builders
# =========================
def build_desc_doc(bio: str) -> str:
    bio = (bio or "").strip()
    if bio == "":
        return ""
    return f"[USER_BIO]\n{bio}\n"

def build_tweet_doc(
    user_tweets_raw,
    M: int = 100,
    K: int = 20,
    max_chars_per_tweet: int = 220,
    total_max_chars: int = 3500
) -> str:
    """
    Per-user structured TweetDoc:
      - [TWEET_BEHAVIOR_SUMMARY] computed on first M tweets
      - [REPRESENTATIVE_TWEETS] up to K cleaned tweets
    """
    raw = [t for t in user_tweets_raw if t is not None]
    raw_M = raw[:M]

    cleaned = [clean_tweet(t, max_chars=max_chars_per_tweet) for t in raw_M]
    cleaned = [t for t in cleaned if t != ""]
    n = len(cleaned)

    if n == 0:
        return (
            "You are analyzing a Twitter user's posting behavior from their tweets.\n\n"
            "[TWEET_BEHAVIOR_SUMMARY]\n"
            "num_tweets_used: 0\n"
            "retweet_rate: 0\n"
            "reply_rate: 0\n"
            "url_rate: 0\n"
            "mention_rate: 0\n"
            "hashtag_rate: 0\n"
            "duplicate_rate: 0\n"
            "avg_tweet_length: 0\n"
            "lexical_diversity: 0\n\n"
            "[REPRESENTATIVE_TWEETS]\n"
            "None\n"
        )

    # retweet rate
    rt = sum(1 for t in cleaned if t.startswith("RT "))

    # reply rate: prefer raw (starts with @)
    reply = 0
    for t_raw in raw_M:
        if t_raw is None:
            continue
        s = t_raw.lstrip()
        if s.startswith("@"):
            reply += 1
    # fallback approximation if raw has no leading @ seen
    if reply == 0:
        reply = sum(1 for t in cleaned if t.startswith("<USER>") and not t.startswith("RT "))

    url = sum(1 for t in cleaned if "<URL>" in t)
    mention = sum(1 for t in cleaned if "<USER>" in t)
    hashtag = sum(1 for t in cleaned if "#" in t)

    normed = [normalize_for_dup(t) for t in raw_M]
    normed = [t for t in normed if t != ""]
    dup_rate = 1.0 - (len(set(normed)) / float(len(normed))) if normed else 0.0

    avg_len = sum(len(t) for t in cleaned) / float(n)
    lex_div = lexical_diversity(cleaned)

    # representative tweets: unique cleaned tweets in order
    rep = []
    seen = set()
    for t in cleaned:
        if t not in seen:
            rep.append(t)
            seen.add(t)
        if len(rep) >= K:
            break
    if len(rep) < K:
        for t in cleaned:
            if len(rep) >= K:
                break
            rep.append(t)

    doc = (
        "You are analyzing a Twitter user's posting behavior from their tweets.\n\n"
        "[TWEET_BEHAVIOR_SUMMARY]\n"
        f"num_tweets_used: {n}\n"
        f"retweet_rate: {rt/n:.3f}\n"
        f"reply_rate: {reply/n:.3f}\n"
        f"url_rate: {url/n:.3f}\n"
        f"mention_rate: {mention/n:.3f}\n"
        f"hashtag_rate: {hashtag/n:.3f}\n"
        f"duplicate_rate: {dup_rate:.3f}\n"
        f"avg_tweet_length: {avg_len:.1f}\n"
        f"lexical_diversity: {lex_div:.3f}\n\n"
        "[REPRESENTATIVE_TWEETS]\n"
    )
    for i, t in enumerate(rep, 1):
        doc += f"{i}) {t}\n"

    if total_max_chars > 0 and len(doc) > total_max_chars:
        doc = doc[:total_max_chars].rstrip() + "\n…\n"
    return doc

# =========================
# Embedding runner
# =========================
@torch.no_grad()
def encode_texts(model: SentenceTransformer, texts, batch_size: int, normalize: bool):
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=normalize
    )
    return emb.float()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Twibot-20")
    parser.add_argument("--processed_dir", type=str, default="./processed_data")

    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--normalize", action="store_true",
                        help="L2 normalize embeddings (useful if you later build semantic kNN).")

    parser.add_argument("--tweet_M", type=int, default=100, help="Max tweets per user used for stats/representation.")
    parser.add_argument("--tweet_K", type=int, default=20, help="Representative tweets kept in TweetDoc.")
    parser.add_argument("--max_chars_per_tweet", type=int, default=220)
    parser.add_argument("--total_max_chars", type=int, default=3500)

    parser.add_argument("--save_structured_text", action="store_true",
                        help="Save structured docs to jsonl for reproducibility.")
    parser.add_argument("--out_tag", type=str, default="enh",
                        help="Output tag. e.g., enh -> des_tensor_enh.pt / tweets_tensor_enh.pt")

    parser.add_argument("--require_dim", type=int, default=768,
                        help="Assert embedding dim for compatibility. Set 0 to disable.")
    args = parser.parse_args()

    ensure_dir(args.processed_dir)

    # ---- load dataset ----
    user, tweet = fast_merge(dataset=args.dataset)
    user_text = list(user["description"])
    tweet_text = [t for t in tweet.text]

    # ---- load mapping ----
    map_path = os.path.join(args.processed_dir, "each_user_tweets.npy")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Cannot find {map_path}. Please ensure it exists.")
    each_user_tweets = torch.load(map_path)

    # ---- load sentence encoder ----
    model = SentenceTransformer(args.model, device=args.device)
    emb_dim = model.get_sentence_embedding_dimension()
    if args.require_dim and emb_dim != args.require_dim:
        raise ValueError(
            f"Embedding dim mismatch: got {emb_dim}, require {args.require_dim}. "
            f"Choose a 768-dim model (e.g., sentence-transformers/all-mpnet-base-v2) "
            f"or set --require_dim 0."
        )

    # =========================
    # 1) Description embeddings
    # =========================
    des_path = os.path.join(args.processed_dir, f"des_tensor_{args.out_tag}.pt")
    structured_desc_path = os.path.join(args.processed_dir, f"structured_desc_{args.out_tag}.jsonl")

    if os.path.exists(des_path):
        print(f"[Desc] Found existing: {des_path} (skip)")
    else:
        print(f"[Desc] Encoding descriptions with {args.model}")
        N = len(user_text)
        des_tensor = torch.zeros((N, emb_dim), dtype=torch.float32)

        fout = open(structured_desc_path, "w", encoding="utf-8") if args.save_structured_text else None

        batch_texts, batch_idx = [], []
        for i in tqdm(range(N), desc="Desc"):
            doc = build_desc_doc(user_text[i])

            if fout is not None:
                fout.write(json.dumps({"user_index": i, "doc": doc}, ensure_ascii=False) + "\n")

            if doc == "":
                # keep zero vector for missing bio
                continue

            batch_texts.append(doc)
            batch_idx.append(i)

            if len(batch_texts) >= args.batch_size:
                emb = encode_texts(model, batch_texts, args.batch_size, args.normalize)
                des_tensor[torch.tensor(batch_idx, dtype=torch.long)] = emb.cpu()
                batch_texts, batch_idx = [], []

        if batch_texts:
            emb = encode_texts(model, batch_texts, args.batch_size, args.normalize)
            des_tensor[torch.tensor(batch_idx, dtype=torch.long)] = emb.cpu()

        if fout is not None:
            fout.close()

        torch.save(des_tensor, des_path)
        print(f"[Desc] Saved: {des_path}")
        if args.save_structured_text:
            print(f"[Desc] Structured docs saved: {structured_desc_path}")

    # =========================
    # 2) Tweet embeddings
    # =========================
    tweets_path = os.path.join(args.processed_dir, f"tweets_tensor_{args.out_tag}.pt")
    structured_tweets_path = os.path.join(args.processed_dir, f"structured_tweets_{args.out_tag}.jsonl")

    if os.path.exists(tweets_path):
        print(f"[Tweet] Found existing: {tweets_path} (skip)")
        return

    print(f"[Tweet] Encoding structured TweetDoc with {args.model}")
    U = len(each_user_tweets)
    tweet_tensor = torch.zeros((U, emb_dim), dtype=torch.float32)

    fout = open(structured_tweets_path, "w", encoding="utf-8") if args.save_structured_text else None

    batch_texts, batch_idx = [], []
    for i in tqdm(range(U), desc="TweetDoc"):
        idxs = each_user_tweets[i]

        if len(idxs) == 0:
            # no tweets -> keep zero vector
            doc = build_tweet_doc([], M=args.tweet_M, K=args.tweet_K,
                                  max_chars_per_tweet=args.max_chars_per_tweet,
                                  total_max_chars=args.total_max_chars)
            if fout is not None:
                fout.write(json.dumps({"user_index": i, "doc": doc}, ensure_ascii=False) + "\n")
            continue

        user_tweets_raw = []
        for j in idxs[:args.tweet_M]:
            user_tweets_raw.append(tweet_text[int(j)])

        doc = build_tweet_doc(
            user_tweets_raw,
            M=args.tweet_M,
            K=args.tweet_K,
            max_chars_per_tweet=args.max_chars_per_tweet,
            total_max_chars=args.total_max_chars
        )

        if fout is not None:
            fout.write(json.dumps({"user_index": i, "doc": doc}, ensure_ascii=False) + "\n")

        batch_texts.append(doc)
        batch_idx.append(i)

        if len(batch_texts) >= args.batch_size:
            emb = encode_texts(model, batch_texts, args.batch_size, args.normalize)
            tweet_tensor[torch.tensor(batch_idx, dtype=torch.long)] = emb.cpu()
            batch_texts, batch_idx = [], []

    if batch_texts:
        emb = encode_texts(model, batch_texts, args.batch_size, args.normalize)
        tweet_tensor[torch.tensor(batch_idx, dtype=torch.long)] = emb.cpu()

    if fout is not None:
        fout.close()

    torch.save(tweet_tensor, tweets_path)
    print(f"[Tweet] Saved: {tweets_path}")
    if args.save_structured_text:
        print(f"[Tweet] Structured docs saved: {structured_tweets_path}")

if __name__ == "__main__":
    main()
