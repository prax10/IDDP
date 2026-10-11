"""
Task M2: comment TEXT features, strictly before each checkpoint moment T.

Input: data/raw/comment_text.csv.gz  (Issue_ID, Creation_Date, <text column>) exported from TAWOS_DB.Comment.
Per comment (computed once):
  * TF-IDF (1-2 grams, 30k vocab) -> TruncatedSVD(24). Both FITTED ONLY ON COMMENTS WRITTEN BEFORE
    THE VALIDATION PERIOD STARTS, so no later text shapes the representation.
  * keyword flags: patch / pull request / merged / committed; blocked / waiting / depends;
    cannot reproduce / need more info; workaround; question mark; "fixed" / "resolved"; length.
Per (issue, T), using only comments with time < T (cumulative sums over each issue's sorted comments):
  ctext_mean_k   mean SVD vector of all earlier comments   (24)
  ctext_last_k   SVD vector of the latest earlier comment  (24)
  ckw_*          counts of each keyword family so far, and in the latest comment
  c_len_mean, c_len_last
Output: data/features/m2_comment_text_features.parquet keyed by (Issue_ID, checkpoint_index).
"""
import re
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from feature_engine import secs, SCALE

K = 24
KW = {"patch": r"\bpatch|pull request|\bpr\b|github\.com/\S+/pull|gerrit|merged|\bcommit(?:ted)?\b|\bpushed\b",
      "blocked": r"\bblock(?:ed|ing|er)\b|\bwaiting\b|\bdepends on\b|\bon hold\b",
      "cantrepro": r"can(?:no|')t reproduce|unable to reproduce|need(?:s)? more info|more information",
      "workaround": r"\bworkaround\b",
      "question": r"\?",
      "fixed": r"\bfixed\b|\bresolved\b|\bdone\b"}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/comment_text.csv.gz"
    t0 = time.time()
    cm = pd.read_csv(path)
    text_col = [c for c in cm.columns if c not in ("Issue_ID", "Creation_Date", "ID", "Author_ID")][0]
    cm = cm.rename(columns={text_col: "text"})
    cm["text"] = cm["text"].fillna("").astype(str).str.slice(0, 3000)   # cap very long pastes (logs) for memory
    cm["t"] = secs(cm["Creation_Date"])
    cm = cm[cm["t"] > 0].sort_values(["Issue_ID", "t"], kind="mergesort").reset_index(drop=True)
    print(f"{len(cm):,} comments on {cm.Issue_ID.nunique():,} issues ({time.time()-t0:.0f}s)", flush=True)

    # fit the text representation on comments written before the validation period starts
    sp = pd.read_csv("data/features/phase8_split.csv")
    val_start = secs(sp.loc[sp.split == "val", "Creation_Date"]).min()
    fit_txt = cm.loc[cm["t"] < val_start, "text"]
    fit_txt = fit_txt.sample(min(len(fit_txt), 400_000), random_state=0)
    vec = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                          strip_accents="unicode", lowercase=True, dtype=np.float32)
    svd = TruncatedSVD(K, random_state=0)
    svd.fit(vec.fit_transform(fit_txt))
    print(f"TF-IDF+SVD fitted on {len(fit_txt):,} pre-validation comments; "
          f"explained variance {svd.explained_variance_ratio_.sum():.2f} ({time.time()-t0:.0f}s)", flush=True)
    # encode in chunks and drop the raw text as soon as possible (memory)
    del fit_txt
    V = np.empty((len(cm), K), np.float32)
    flags = {k: np.empty(len(cm), np.int32) for k in KW}
    length = np.empty(len(cm), np.float32)
    pats = {k: re.compile(p) for k, p in KW.items()}
    for i in range(0, len(cm), 100_000):
        txt = cm["text"].iloc[i:i + 100_000]
        V[i:i + len(txt)] = svd.transform(vec.transform(txt))
        low = txt.str.lower()
        for k, p in pats.items():
            flags[k][i:i + len(txt)] = low.str.contains(p).to_numpy(np.int32)
        length[i:i + len(txt)] = np.log1p(txt.str.len().to_numpy(np.float32))
    cm = cm.drop(columns=["text", "Creation_Date"])
    import gc; gc.collect()
    print(f"comment vectors + keyword flags done ({time.time()-t0:.0f}s)", flush=True)

    # cumulative sums per sorted comment stream
    iid = cm["Issue_ID"].to_numpy(np.int64)
    keys = iid * SCALE + cm["t"].to_numpy(np.int64)
    cumV = np.vstack([np.zeros((1, K), np.float64), np.cumsum(V, axis=0, dtype=np.float64)])
    cumF = {k: np.concatenate([[0], np.cumsum(v, dtype=np.int64)]) for k, v in flags.items()}
    cumL = np.concatenate([[0], np.cumsum(length, dtype=np.float64)])

    ck = pd.read_csv("data/features/phase8_checkpoints.csv", usecols=["Issue_ID", "checkpoint_index", "elapsed_minutes"])
    iss = pd.read_csv("data/raw/issue_slim.csv.gz", usecols=["ID", "Creation_Date"])
    c_s = pd.Series(secs(iss["Creation_Date"]), index=iss["ID"])
    T = c_s.reindex(ck["Issue_ID"]).to_numpy(np.int64) + np.floor(ck["elapsed_minutes"].to_numpy() * 60).astype(np.int64)
    q_iid = ck["Issue_ID"].to_numpy(np.int64)
    hi = np.searchsorted(keys, q_iid * SCALE + T, side="left")      # strictly before T
    lo = np.searchsorted(keys, q_iid * SCALE, side="left")
    n = hi - lo
    has = n > 0
    last = np.clip(hi - 1, 0, len(cm) - 1)
    out = {"Issue_ID": q_iid, "checkpoint_index": ck["checkpoint_index"].to_numpy()}
    mean = (cumV[hi] - cumV[lo]) / np.maximum(n, 1)[:, None]
    for k in range(K):
        out[f"ctext_mean_{k}"] = np.where(has, mean[:, k], np.nan).astype(np.float32)
        out[f"ctext_last_{k}"] = np.where(has, V[last, k], np.nan).astype(np.float32)
    for k in KW:
        out[f"ckw_{k}_n"] = (cumF[k][hi] - cumF[k][lo]).astype(np.int32)
        out[f"ckw_{k}_last"] = np.where(has, flags[k][last], 0).astype(np.int8)
    out["c_len_mean"] = np.where(has, (cumL[hi] - cumL[lo]) / np.maximum(n, 1), np.nan).astype(np.float32)
    out["c_len_last"] = np.where(has, length[last], np.nan).astype(np.float32)
    out = pd.DataFrame(out)
    out.to_parquet("data/features/m2_comment_text_features.parquet", index=False)
    print(f"features {out.shape}; rows with >=1 earlier comment {has.mean():.1%} ({time.time()-t0:.0f}s)")
    import joblib
    joblib.dump({"vectorizer": vec, "svd": svd, "keywords": KW}, "models/m2_comment_text_encoder.joblib")


if __name__ == "__main__":
    main()
