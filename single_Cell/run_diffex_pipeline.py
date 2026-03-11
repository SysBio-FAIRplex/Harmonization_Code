#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from helpers.diffex_helpers import (
    build_group_file_catalog,
    infer_group_info_from_counts_path,
    is_allowed_pair,
    normalize_state,
    read_counts_samples,
    read_table_auto,
    resolve_side_paths,
    run_normalized_linear_model,
    select_side_ids,
)

# ---------------------------
# CONFIG
# ---------------------------
ROOT = Path(".")
COMPARISON_FILE = ROOT / "beta1/pseudobulk_out/1B_comparisonsforDiffex.tsv"
OUT_CONSOLIDATED = ROOT / "beta1/pseudobulk_out/1A_ALL_diffex_consolidated_recomputed_new.tsv"
OUT_SUMMARY = ROOT / "beta1/pseudobulk_out/1A_ALL_diffex_recomputed_summary_new.tsv"

STATE_COL = "case_control"
CASE_SET = {"case", "disease", "positive", "1", "yes", "treated"}
CTRL_SET = {"control", "healthy", "negative", "0", "no", "untreated"}

# Comparison policy requested: use case_control and allow CASE/CONTROL + CASE/CASE.
ALLOW_CASE_CONTROL = True
ALLOW_CASE_CASE = True

MIN_REPS_PER_GROUP = 2

MIN_COUNT_PER_GENE = 10
MIN_SAMPLES_WITH_MIN_COUNT = 3
MIN_CPM_PER_GENE = 1.0
MIN_SAMPLES_WITH_MIN_CPM = 3
MIN_LOGCPM_VAR = 1e-6
MIN_NONZERO_FRACTION_ANY_GROUP = 0.25
MIN_NONZERO_FRACTION_OVERALL = 0.2
MIN_GROUP_MEAN_LOGCPM = 0.75


def run_one_comparison(
    row: pd.Series, group_file_catalog: Dict[Tuple[str, str, str, str], Tuple[str, str, str]]
) -> Tuple[pd.DataFrame, Dict]:
    cid = str(row["comparison_id"])
    summary = {"comparison_id": cid, "status": "not_run", "reason": "", "nA": 0, "nB": 0}

    try:
        counts_a_path, samples_a_path, join_a = resolve_side_paths(row, "A", ROOT, group_file_catalog)
        c_a, s_a, qc_a = read_counts_samples(counts_a_path, samples_a_path, join_a, STATE_COL, CASE_SET, CTRL_SET)

        counts_b_path, samples_b_path, join_b = resolve_side_paths(row, "B", ROOT, group_file_catalog)
        c_b, s_b, _qc_b = read_counts_samples(counts_b_path, samples_b_path, join_b, STATE_COL, CASE_SET, CTRL_SET)

        inferred = infer_group_info_from_counts_path(counts_a_path)
        summary.update(
            {
                "cohort_from_path": inferred["cohort_from_path"],
                "tissue_from_path": inferred["tissue_from_path"],
                "cell_from_path": inferred["cell_from_path"],
                "fraction_integer_like": qc_a["fraction_integer_like"],
                "max_value": qc_a["max_value"],
                "p99": qc_a["p99"],
                "sample_skewness": qc_a["sample_skewness"],
                "sample_excess_kurtosis": qc_a["sample_excess_kurtosis"],
                "likely_transformed": qc_a["likely_transformed"],
            }
        )

        state_a = normalize_state(str(row["stateA"]), CASE_SET, CTRL_SET)
        state_b = normalize_state(str(row["stateB"]), CASE_SET, CTRL_SET)

        if not is_allowed_pair(state_a, state_b, ALLOW_CASE_CONTROL, ALLOW_CASE_CASE):
            summary["status"] = "skipped"
            summary["reason"] = "comparison_state_pair_not_allowed"
            return pd.DataFrame(), summary

        row_local = row.copy()
        row_local["stateA"] = state_a
        row_local["stateB"] = state_b
        ids_a = select_side_ids(s_a, row_local, "A", MIN_REPS_PER_GROUP)
        ids_b = select_side_ids(s_b, row_local, "B", MIN_REPS_PER_GROUP)

        summary["nA"] = len(ids_a)
        summary["nB"] = len(ids_b)
        if len(ids_a) < MIN_REPS_PER_GROUP or len(ids_b) < MIN_REPS_PER_GROUP:
            summary["status"] = "skipped"
            summary["reason"] = "insufficient_replicates_or_metadata"
            return pd.DataFrame(), summary

        common_genes = [g for g in c_a.columns if g in c_b.columns]
        if len(common_genes) == 0:
            summary["status"] = "skipped"
            summary["reason"] = "no_common_genes_between_A_B"
            return pd.DataFrame(), summary

        x_a = c_a.loc[ids_a, common_genes].copy()
        x_b = c_b.loc[ids_b, common_genes].copy()
        c = pd.concat([x_a, x_b], axis=0)

        base, model_qc = run_normalized_linear_model(
            c=c,
            group_a_ids=list(x_a.index),
            group_b_ids=list(x_b.index),
            min_count_per_gene=MIN_COUNT_PER_GENE,
            min_samples_with_min_count=MIN_SAMPLES_WITH_MIN_COUNT,
            min_cpm_per_gene=MIN_CPM_PER_GENE,
            min_samples_with_min_cpm=MIN_SAMPLES_WITH_MIN_CPM,
            min_logcpm_var=MIN_LOGCPM_VAR,
            min_nonzero_fraction_any_group=MIN_NONZERO_FRACTION_ANY_GROUP,
            min_nonzero_fraction_overall=MIN_NONZERO_FRACTION_OVERALL,
            min_group_mean_logcpm=MIN_GROUP_MEAN_LOGCPM,
        )
        summary.update(model_qc)

        if len(base) == 0:
            summary["status"] = "skipped"
            summary["reason"] = "no_genes_after_filtering"
            return pd.DataFrame(), summary

        out = pd.DataFrame(
            {
                "gene": base["gene"],
                "logFC": base["logFC"],
                "tstat": base["tstat"],
                "pval": base["pval"],
                "padj": base["padj"],
                "comparison_id": cid,
                "cohortA": row["cohortA"],
                "cohortB": row["cohortB"],
                "tissue": row["tissueA"],
                "cell": row["cellA"],
                "test_type": row["test_type"],
                "stateA": state_a,
                "stateB": state_b,
                "nA": len(x_a.index),
                "nB": len(x_b.index),
                "source_file": f"{cid}_diffex.tsv",
            }
        )

        summary["status"] = "ok"
        return out, summary

    except Exception as e:
        summary["status"] = "error"
        summary["reason"] = str(e)
        return pd.DataFrame(), summary


def main() -> None:
    print("Using comparisons:", COMPARISON_FILE)
    print("Writing consolidated:", OUT_CONSOLIDATED)
    print("Writing summary:", OUT_SUMMARY)

    comparisons = read_table_auto(COMPARISON_FILE)
    group_file_catalog = build_group_file_catalog(comparisons)

    all_out: List[pd.DataFrame] = []
    summaries: List[Dict] = []

    for _, row in comparisons.iterrows():
        out_df, summary = run_one_comparison(row, group_file_catalog)
        summaries.append(summary)
        if len(out_df) > 0:
            all_out.append(out_df)

    summary_df = pd.DataFrame(summaries)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_SUMMARY, sep="\t", index=False)

    print("Status counts:")
    print(summary_df["status"].value_counts(dropna=False))

    if len(all_out) == 0:
        raise RuntimeError("No comparisons produced results. Inspect summary file.")

    final = pd.concat(all_out, ignore_index=True)
    expected_cols = [
        "gene",
        "logFC",
        "tstat",
        "pval",
        "padj",
        "comparison_id",
        "cohortA",
        "cohortB",
        "tissue",
        "cell",
        "test_type",
        "stateA",
        "stateB",
        "nA",
        "nB",
        "source_file",
    ]
    final = final[expected_cols]
    OUT_CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_CONSOLIDATED, sep="\t", index=False)

    print("Saved consolidated:", OUT_CONSOLIDATED)
    print("Saved summary:", OUT_SUMMARY)
    print("Rows:", len(final))
    print(summary_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()

