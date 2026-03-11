#!/usr/bin/env python3
"""
Simple EDA PCA script for a single counts/samples pair.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from helpers.diffex_helpers import read_counts_samples

ROOT = Path(".")
COUNTS_PATH = ROOT / "beta1/pseudobulk_out/pseudobulk/CMD_pseudobulk_out/kidney/b_cell/counts.tsv"
SAMPLES_PATH = ROOT / "beta1/pseudobulk_out/pseudobulk/CMD_pseudobulk_out/kidney/b_cell/samples.tsv"
JOIN_KEY = "biosample_id"
STATE_COL = "case_control"

CASE_SET = {"case", "disease", "positive", "1", "yes", "treated"}
CTRL_SET = {"control", "healthy", "negative", "0", "no", "untreated"}

OUT_PNG = ROOT / "beta1/pseudobulk_out/pca_example.png"


def main() -> None:
    mat, samples, _qc = read_counts_samples(
        COUNTS_PATH, SAMPLES_PATH, JOIN_KEY, STATE_COL, CASE_SET, CTRL_SET
    )

    lib = mat.sum(axis=1).replace(0, np.nan)
    cpm = mat.div(lib, axis=0) * 1e6
    x = np.log2(cpm + 1.0).fillna(0.0)

    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(x)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=x.index)
    pca_df = pca_df.join(samples.set_index("__sample_id__")[["__state__"]], how="left")

    color_map = {"CASE": "#ef4444", "CONTROL": "#2563eb", "UNKNOWN": "#9ca3af"}
    colors = pca_df["__state__"].fillna("UNKNOWN").map(color_map).fillna("#9ca3af")

    plt.figure(figsize=(7, 6))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], c=colors, alpha=0.75, s=30)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    plt.title("PCA (log2(CPM+1))")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=180)
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()

