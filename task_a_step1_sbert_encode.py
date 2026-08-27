"""
Task A, Step 1: SBERT-encode Title + Description_Text for the 203,290
resolved issues only. Caches raw (203290, 384) embeddings immediately.
"""

import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

RAW_DIR = "data/raw"
FEATURES_DIR = "data/features"
OUT_PATH = f"{FEATURES_DIR}/sbert_raw.npy"
IDS_PATH = f"{FEATURES_DIR}/sbert_raw_ids.npy"


def main():
    print("Loading resolved-issue ID list + text fields ...")
    feat_ids = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table.csv", usecols=["ID"])
    issue = pd.read_csv(
        f"{RAW_DIR}/issue.csv", usecols=["ID", "Title", "Description_Text"]
    )
    issue = issue[issue["ID"].isin(feat_ids["ID"])].copy()
    issue = issue.set_index("ID").loc[feat_ids["ID"]].reset_index()
    assert len(issue) == 203290, f"expected 203290 resolved issues, got {len(issue)}"

    text = issue["Title"].fillna("") + ". " + issue["Description_Text"].fillna("")
    text = text.tolist()
    print(f"  {len(text)} texts prepared")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading SentenceTransformer('all-MiniLM-L6-v2') on device={device} ...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    t0 = time.time()
    embeddings = model.encode(
        text, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    elapsed = time.time() - t0
    print(f"\nEncoding done in {elapsed:.1f}s ({elapsed/60:.1f} min). Shape: {embeddings.shape}")

    np.save(OUT_PATH, embeddings)
    np.save(IDS_PATH, issue["ID"].to_numpy())
    print(f"Saved embeddings -> {OUT_PATH}")
    print(f"Saved matching ID order -> {IDS_PATH}")


if __name__ == "__main__":
    main()
