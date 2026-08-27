"""
Task A, Step 2: PCA-reduce SBERT embeddings to 30 dims, fit on
training-split issues only (leakage-safe), transform all resolved issues.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

FEATURES_DIR = "data/features"


def main():
    embeddings = np.load(f"{FEATURES_DIR}/sbert_raw.npy")
    ids = np.load(f"{FEATURES_DIR}/sbert_raw_ids.npy")
    print(f"Loaded embeddings: {embeddings.shape}")

    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    split_lookup = split.set_index("ID")["split"]
    row_split = pd.Series(ids).map(split_lookup).to_numpy()
    train_mask = row_split == "train"
    print(f"  train rows for PCA fit: {train_mask.sum()} / {len(ids)}")

    pca = PCA(n_components=30, random_state=0)
    pca.fit(embeddings[train_mask])
    emb30 = pca.transform(embeddings)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print("\nCumulative explained variance ratio (30 components):")
    for k in [1, 5, 10, 15, 20, 25, 30]:
        print(f"  first {k:2d} components: {cum_var[k-1]:.4f}")

    cols = [f"text_pc_{i}" for i in range(30)]
    out_df = pd.DataFrame(emb30, columns=cols)
    out_df.insert(0, "ID", ids)
    out_path = f"{FEATURES_DIR}/text_pca30.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
