from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================================================
# SELECT FILES
# =========================================================
h5_files = [
    "beta1/pseudobulk_out/pseudobulk/CMD_pseudobulk_out/liver/b/pseudobulk.h5",
    "beta1/pseudobulk_out/pseudobulk/CMD_pseudobulk_out/liver/cholangiocyte/pseudobulk.h5",
    "beta1/pseudobulk_out/pseudobulk/CMD_pseudobulk_out/liver/endothelial/pseudobulk.h5",
    "beta1/pseudobulk_out/pseudobulk/CMD_pseudobulk_out/liver/hepatocytes/pseudobulk.h5",
    "beta1/pseudobulk_out/pseudobulk/CMD_pseudobulk_out/liver/myeloid/pseudobulk.h5",
]

color_by = "source_file"   # try: source_file, tissue, cell_type, sample_id
scale_genes = True
min_detect_frac = 0.20
min_total_count = 10



# =========================================================
# HELPERS
# =========================================================
def decode_arr(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.array(out, dtype=object)



def load_pseudobulk_h5(fp):
    fp = Path(fp)

    with h5py.File(fp, "r") as f:
        X = f["X"][:]
        sample_ids = decode_arr(f["sample_ids"][:])
        gene_ids = decode_arr(f["gene_ids"][:])

        obs = pd.DataFrame(index=sample_ids)

        if "samples" in f:
            for col in f["samples"].keys():
                vals = f["samples"][col][:]
                if getattr(vals, "dtype", None) is not None and vals.dtype.kind in {"S", "O", "U"}:
                    obs[col] = decode_arr(vals)
                else:
                    obs[col] = vals

        obs["source_file"] = fp.parent.name
        obs["source_path"] = str(fp)

        if "metadata" in f:
            meta = f["metadata"].attrs
            if "tissue" in meta:
                obs["tissue"] = str(meta["tissue"])
            if "cell_type" in meta:
                obs["cell_type"] = str(meta["cell_type"])

    return {
        "X": X,
        "sample_ids": sample_ids,
        "gene_ids": gene_ids,
        "obs": obs,
        "file": str(fp),
    }



def collapse_duplicate_genes(item):
    """
    If gene_ids are duplicated, collapse them by summing columns.
    Keeps samples as rows.
    """
    X = item["X"]
    gene_ids = pd.Index(item["gene_ids"].astype(str))

    if gene_ids.is_unique:
        item["gene_ids"] = gene_ids.to_numpy(dtype=object)
        return item

    print(f"Duplicate genes detected in {item['file']}")
    print(f"  original genes: {len(gene_ids)}")
    print(f"  unique genes:    {gene_ids.nunique()}")

    df = pd.DataFrame(X, index=item["sample_ids"], columns=gene_ids)
    df = df.groupby(level=0, axis=1).sum()

    item["X"] = df.to_numpy()
    item["gene_ids"] = df.columns.astype(str).to_numpy(dtype=object)
    return item



def intersect_genes(loaded):
    shared = set(loaded[0]["gene_ids"])
    for item in loaded[1:]:
        shared &= set(item["gene_ids"])
    return sorted(shared)



def subset_to_genes(item, shared_genes):
    gene_index = pd.Index(item["gene_ids"].astype(str))
    shared_mask = gene_index.isin(shared_genes)

    X_sub = item["X"][:, shared_mask]
    genes_sub = gene_index[shared_mask]

    # reorder exactly to shared_genes
    df = pd.DataFrame(X_sub, index=item["obs"].index, columns=genes_sub)
    df = df.loc[:, shared_genes]

    return df.to_numpy()



def preprocess_matrix(X):
    detected_frac = (X > 0).sum(axis=0) / X.shape[0]
    total_counts = X.sum(axis=0)

    keep = (detected_frac >= min_detect_frac) & (total_counts >= min_total_count)
    X = X[:, keep]

    X = np.log1p(X)

    keep_var = X.var(axis=0) > 0
    X = X[:, keep_var]

    if scale_genes:
        X = StandardScaler().fit_transform(X)

    return X



# =========================================================
# LOAD + FIX DUPLICATES
# =========================================================
loaded = [load_pseudobulk_h5(fp) for fp in h5_files]
loaded = [collapse_duplicate_genes(item) for item in loaded]
shared_genes = intersect_genes(loaded)

# =========================================================
# COMBINE
# =========================================================
X_list = []
obs_list = []

for item in loaded:
    X_sub = subset_to_genes(item, shared_genes)
    obs = item["obs"].copy()

    obs.index = [f"{Path(item['file']).parent.name}__{idx}" for idx in obs.index]

    X_list.append(X_sub)
    obs_list.append(obs)

X = np.vstack(X_list)
obs = pd.concat(obs_list, axis=0)

X_pca = preprocess_matrix(X)

# =========================================================
# PCA
# =========================================================
pca = PCA(n_components=5)
scores = pca.fit_transform(X_pca)

plot_df = obs.copy()
plot_df["PC1"] = scores[:, 0]
plot_df["PC2"] = scores[:, 1]

for i, v in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"PC{i}: {v*100:.2f}%")



# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(9, 7))

if color_by in plot_df.columns:
    for label, subdf in plot_df.groupby(color_by):
        plt.scatter(subdf["PC1"], subdf["PC2"], alpha=0.8, label=str(label))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
else:
    plt.scatter(plot_df["PC1"], plot_df["PC2"], alpha=0.8)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("PCA Across Selected HDF5 Files")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

if color_by in plot_df.columns:
    print(f"\nCounts by {color_by}:")
    print(plot_df[color_by].value_counts())