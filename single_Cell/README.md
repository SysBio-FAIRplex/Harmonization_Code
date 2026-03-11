# SysBIO FAIRplex Single Cell (GitHub Ready)

Clean folder layout with:

- main DiffEx pipeline runner: `run_diffex_pipeline.py`
- helper functions: `helpers/diffex_helpers.py`
- exploratory scripts: `EDA/`
- diagram assets: `diagram/`

## Pipeline map (Mermaid)

```mermaid
flowchart LR
    A["1) Cohorts<br/>AMP-RA/SLE, AMP-CMD, AMP-PD, AMP-AD"] --> B["2) Pseudobulk"]
    B --> C["3) Harmonize"]
    C --> D["4) QC"]
    D --> E["5) DiffEx"]
    E --> F1["DE"]
    E --> F2["Volcano plots"]
    D --> F3["PCA"]
    D --> F4["QC"]
```

## DiffEx policy

The main script uses `case_control` and currently allows:

- `CASE vs CONTROL`
- `CASE vs CASE`

This behavior is configured in `run_diffex_pipeline.py`:

- `ALLOW_CASE_CONTROL = True`
- `ALLOW_CASE_CASE = True`

## Run DiffEx

From repository root:

```bash
python3 single_Cell/run_diffex_pipeline.py
```

## Run EDA PCA example

```bash
python3 single_Cell/EDA/pca_from_counts.py
```

## Diagram source

- Metro map definition: `diagram/sysbio_fairplex_pipeline.mmd`
- To render SVG:

```bash
nf-metro render single_Cell/diagram/sysbio_fairplex_pipeline.mmd -o single_Cell/diagram/sysbio_fairplex_pipeline.svg --theme light
```

