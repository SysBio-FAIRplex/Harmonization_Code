from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


GROUP_FIELD_CANDIDATES = {
    "cohort": ["cohort", "dataset", "study", "source_cohort"],
    "tissue": ["tissue", "organ", "region", "brain_region"],
    "cell": ["cell", "cell_type", "celltype", "cluster", "annotation"],
}


def read_table_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1 and "\t" in str(df.columns[0]):
        df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def first_matching_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def norm_label(x: str) -> str:
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def group_key(cohort: str, tissue: str, cell: str, state: str) -> Tuple[str, str, str, str]:
    return (norm_label(cohort), norm_label(tissue), norm_label(cell), norm_label(state))


def normalize_state(x: str, case_set: set[str], ctrl_set: set[str]) -> str:
    v = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    if v in case_set:
        return "CASE"
    if v in ctrl_set:
        return "CONTROL"
    if v in {"", "na", "n/a", "none", "null", "unknown", "nan"}:
        return "UNKNOWN"
    return str(x).strip().upper()


def is_allowed_pair(state_a: str, state_b: str, allow_case_control: bool, allow_case_case: bool) -> bool:
    pair = {state_a, state_b}
    if allow_case_control and pair == {"CASE", "CONTROL"}:
        return True
    if allow_case_case and state_a == "CASE" and state_b == "CASE":
        return True
    return False


def assess_count_scale(mat: pd.DataFrame) -> Dict[str, float]:
    vals = mat.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {
            "fraction_integer_like": 0.0,
            "max_value": np.nan,
            "p99": np.nan,
            "sample_skewness": np.nan,
            "sample_excess_kurtosis": np.nan,
            "likely_transformed": True,
        }
    frac_int = float(np.mean(np.isclose(finite, np.round(finite), atol=1e-8)))
    max_v = float(np.max(finite))
    p99 = float(np.percentile(finite, 99))
    sample = finite[: min(len(finite), 200000)]
    sample_skew = float(stats.skew(sample, bias=False, nan_policy="omit"))
    sample_kurt = float(stats.kurtosis(sample, fisher=True, bias=False, nan_policy="omit"))
    likely_transformed = (frac_int < 0.95) or (max_v < 50 and p99 < 20)
    return {
        "fraction_integer_like": frac_int,
        "max_value": max_v,
        "p99": p99,
        "sample_skewness": sample_skew,
        "sample_excess_kurtosis": sample_kurt,
        "likely_transformed": bool(likely_transformed),
    }


def infer_group_info_from_counts_path(counts_path: Path) -> Dict[str, str]:
    parts = list(counts_path.parts)
    lower_parts = [p.lower() for p in parts]

    cohort = "UNKNOWN"
    tissue = "UNKNOWN"
    cell = "UNKNOWN"

    idx = None
    for i, p in enumerate(lower_parts):
        if "pseudobulk_out" in p:
            idx = i
            break

    if idx is not None:
        cohort_raw = parts[idx]
        cohort = cohort_raw.replace("_pseudobulk_out", "").replace("pseudobulk_out", "").strip("_")
        if idx + 1 < len(parts):
            tissue = parts[idx + 1]
        if idx + 2 < len(parts):
            maybe = parts[idx + 2]
            if maybe.lower().startswith("counts") or "." in maybe:
                maybe = counts_path.stem
            cell = maybe

    if cell == "UNKNOWN":
        cell = counts_path.stem

    cell = (
        cell.replace("_pseudobulk_counts", "")
        .replace("_counts", "")
        .replace("counts", "")
        .strip("_")
    ) or "UNKNOWN"

    return {"cohort_from_path": cohort, "tissue_from_path": tissue, "cell_from_path": cell}


def read_counts_samples(
    counts_path: Path,
    samples_path: Path,
    join_key: str,
    state_col: str,
    case_set: set[str],
    ctrl_set: set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    counts = read_table_auto(counts_path)
    samples = read_table_auto(samples_path)

    if counts.shape[1] < 2:
        raise ValueError(f"Malformed counts: {counts_path}")

    first_col = counts.columns[0]
    if first_col.lower() in {"gene", "gene_id", "feature", "symbol", "id"} or not pd.api.types.is_numeric_dtype(
        counts[first_col]
    ):
        counts = counts.set_index(first_col)

    counts.index = counts.index.astype(str).str.strip()
    counts.columns = counts.columns.astype(str).str.strip()

    if join_key not in samples.columns:
        cols = {c.lower(): c for c in samples.columns}
        if join_key.lower() in cols:
            join_key = cols[join_key.lower()]
        else:
            raise KeyError(f"join_key '{join_key}' not found in {samples_path}")

    sample_ids = samples[join_key].astype(str).str.strip()
    overlap_cols = len(set(counts.columns) & set(sample_ids))
    overlap_idx = len(set(counts.index) & set(sample_ids))
    mat = counts.T if overlap_cols >= overlap_idx else counts

    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0)
    mat.index = mat.index.astype(str).str.strip()
    mat.columns = mat.columns.astype(str).str.strip()

    if mat.index.has_duplicates:
        mat = mat.groupby(level=0, sort=False).sum()
    if mat.columns.has_duplicates:
        mat = mat.T.groupby(level=0, sort=False).sum().T

    scale_qc = assess_count_scale(mat)
    mat = np.round(mat).astype(int)

    samples = samples.copy()
    samples["__sample_id__"] = samples[join_key].astype(str).str.strip()

    state_col_found = None
    for c in samples.columns:
        if str(c).strip().lower() == state_col.lower():
            state_col_found = c
            break
    if state_col_found is None:
        raise KeyError(f"Required state column '{state_col}' not found in {samples_path}")

    samples["__state__"] = samples[state_col_found].map(lambda x: normalize_state(x, case_set, ctrl_set))

    cohort_col = first_matching_col(list(samples.columns), GROUP_FIELD_CANDIDATES["cohort"])
    tissue_col = first_matching_col(list(samples.columns), GROUP_FIELD_CANDIDATES["tissue"])
    cell_col = first_matching_col(list(samples.columns), GROUP_FIELD_CANDIDATES["cell"])

    if cohort_col:
        samples["__cohort__"] = samples[cohort_col].astype(str).str.strip()
    if tissue_col:
        samples["__tissue__"] = samples[tissue_col].astype(str).str.strip()
    if cell_col:
        samples["__cell__"] = samples[cell_col].astype(str).str.strip()

    samples = samples.drop_duplicates(subset=["__sample_id__"], keep="first")
    samples = samples[samples["__sample_id__"].isin(mat.index)].copy()
    mat = mat.loc[samples["__sample_id__"]]

    return mat, samples, scale_qc


def build_group_file_catalog(comparisons_df: pd.DataFrame) -> Dict[Tuple[str, str, str, str], Tuple[str, str, str]]:
    cat: Dict[Tuple[str, str, str, str], Tuple[str, str, str]] = {}
    for _, r in comparisons_df.iterrows():
        key_a = group_key(str(r["cohortA"]), str(r["tissueA"]), str(r["cellA"]), str(r["stateA"]))
        if key_a not in cat:
            cat[key_a] = (str(r["counts_path"]), str(r["samples_path"]), str(r["join_key"]))
    return cat


def resolve_side_paths(
    row: pd.Series, side: str, root: Path, cat: Dict[Tuple[str, str, str, str], Tuple[str, str, str]]
) -> Tuple[Path, Path, str]:
    if side == "A":
        return root / str(row["counts_path"]), root / str(row["samples_path"]), str(row["join_key"])

    same_group = (
        norm_label(row["cohortA"]) == norm_label(row["cohortB"])
        and norm_label(row["tissueA"]) == norm_label(row["tissueB"])
        and norm_label(row["cellA"]) == norm_label(row["cellB"])
        and norm_label(row["stateA"]) == norm_label(row["stateB"])
    )
    if same_group:
        return root / str(row["counts_path"]), root / str(row["samples_path"]), str(row["join_key"])

    key_b = group_key(str(row["cohortB"]), str(row["tissueB"]), str(row["cellB"]), str(row["stateB"]))
    if key_b not in cat:
        raise KeyError(f"No B-side mapping found for {key_b}. Need this group to appear as A-side in comparisons.")
    c_path, s_path, j_key = cat[key_b]
    return root / c_path, root / s_path, j_key


def select_side_ids(samples: pd.DataFrame, row: pd.Series, side: str, min_reps_per_group: int) -> List[str]:
    state_key = f"state{side}"
    cohort_key = f"cohort{side}"
    tissue_key = f"tissue{side}"
    cell_key = f"cell{side}"

    wanted_state = str(row[state_key]).upper()
    mask = samples["__state__"].astype(str).str.upper() == wanted_state

    for sample_col, row_key in [
        ("__cohort__", cohort_key),
        ("__tissue__", tissue_key),
        ("__cell__", cell_key),
    ]:
        if sample_col not in samples.columns:
            continue
        target = str(row.get(row_key, "")).strip()
        if target == "" or target.lower() in {"na", "n/a", "none", "null", "nan"}:
            continue

        exact = mask & (samples[sample_col].astype(str).str.upper() == target.upper())
        if exact.sum() >= min_reps_per_group:
            mask = exact
            continue

        fuzzy = mask & (
            samples[sample_col].astype(str).map(norm_label).str.contains(norm_label(target), regex=False)
        )
        if fuzzy.sum() >= min_reps_per_group:
            mask = fuzzy

    return samples.loc[mask, "__sample_id__"].tolist()


def run_normalized_linear_model(
    c: pd.DataFrame,
    group_a_ids: List[str],
    group_b_ids: List[str],
    min_count_per_gene: int,
    min_samples_with_min_count: int,
    min_cpm_per_gene: float,
    min_samples_with_min_cpm: int,
    min_logcpm_var: float,
    min_nonzero_fraction_any_group: float,
    min_nonzero_fraction_overall: float,
    min_group_mean_logcpm: float,
) -> Tuple[pd.DataFrame, Dict]:
    ids = group_a_ids + group_b_ids
    y_counts = c.loc[ids].copy()
    qc = {"genes_input": int(y_counts.shape[1])}

    keep_count = (y_counts >= min_count_per_gene).sum(axis=0) >= min_samples_with_min_count
    lib0 = y_counts.sum(axis=1).replace(0, np.nan)
    cpm0 = y_counts.div(lib0, axis=0) * 1e6
    keep_cpm = (cpm0 >= min_cpm_per_gene).sum(axis=0) >= min_samples_with_min_cpm
    y_counts = y_counts.loc[:, keep_count | keep_cpm]
    qc["genes_after_count_cpm_filter"] = int(y_counts.shape[1])

    if y_counts.shape[1] == 0:
        qc["genes_after_zero_expr_filter"] = 0
        qc["genes_after_var_filter"] = 0
        return pd.DataFrame(columns=["gene", "logFC", "tstat", "pval", "padj"]), qc

    lib = y_counts.sum(axis=1).replace(0, np.nan)
    cpm = y_counts.div(lib, axis=0) * 1e6
    y = np.log2(cpm + 1.0).fillna(0.0)

    y_a = y.loc[group_a_ids]
    y_b = y.loc[group_b_ids]
    nonzero_a = (y_a > 0.0).mean(axis=0)
    nonzero_b = (y_b > 0.0).mean(axis=0)
    nonzero_overall = (y > 0.0).mean(axis=0)
    mean_a = y_a.mean(axis=0)
    mean_b = y_b.mean(axis=0)

    keep_nonzero = (
        ((nonzero_a >= min_nonzero_fraction_any_group) | (nonzero_b >= min_nonzero_fraction_any_group))
        & (nonzero_overall >= min_nonzero_fraction_overall)
    )
    keep_expr = (mean_a >= min_group_mean_logcpm) | (mean_b >= min_group_mean_logcpm)
    y = y.loc[:, keep_nonzero & keep_expr]
    qc["genes_after_zero_expr_filter"] = int(y.shape[1])

    y = y.loc[:, y.var(axis=0) > min_logcpm_var]
    qc["genes_after_var_filter"] = int(y.shape[1])

    if y.shape[1] == 0:
        return pd.DataFrame(columns=["gene", "logFC", "tstat", "pval", "padj"]), qc

    x = pd.DataFrame(index=ids)
    x["Intercept"] = 1.0
    x["groupA"] = [1.0] * len(group_a_ids) + [0.0] * len(group_b_ids)

    xv = x.to_numpy(float)
    yv = y.to_numpy(float)
    n, p = xv.shape

    xtx_inv = np.linalg.pinv(xv.T @ xv)
    b = xtx_inv @ xv.T @ yv
    resid = yv - (xv @ b)

    df_resid = max(n - p, 1)
    sigma2 = (resid**2).sum(axis=0) / df_resid

    gidx = list(x.columns).index("groupA")
    se = np.sqrt(np.maximum(sigma2 * xtx_inv[gidx, gidx], 1e-12))
    tstat = b[gidx, :] / se
    pval = 2 * stats.t.sf(np.abs(tstat), df=df_resid)
    padj = multipletests(pval, method="fdr_bh")[1]

    out = pd.DataFrame(
        {"gene": y.columns, "logFC": b[gidx, :], "tstat": tstat, "pval": pval, "padj": padj}
    )
    out["pval"] = np.clip(out["pval"].astype(float), 1e-300, 1.0)
    out["padj"] = np.clip(out["padj"].astype(float), 1e-300, 1.0)
    return out, qc
