# Privacy-preserving AI agent for clinical calculators

Six generic notebooks, one per strategy. Each has a **Configuration** cell (dataset, model, paths) and an **Evaluation protocol** section that states the metric definitions, denominators, thresholds, decoding settings and error rules used by that notebook.

| File | Strategy |
|---|---|
| `notebooks/01_mcp_pipeline.ipynb` | MCP (proposed): registry-based selection, schema-guided extraction, deterministic computation on the MCP server |
| `notebooks/02_zero_shot_direct.ipynb` | Zero-Shot Direct |
| `notebooks/03_zero_shot_cot.ipynb` | Zero-Shot Chain-of-Thought |
| `notebooks/04_one_shot_cot.ipynb` | One-Shot Chain-of-Thought |
| `notebooks/05_tool_calling.ipynb` | Tool Calling (calculator code is executed) |
| `notebooks/06_rag_ci.ipynb` | RAG + CI |

## Repository layout

```
notebooks/                six notebooks (one per strategy)
registry/
  mcp_registry_abacus.json    MCP registry, 68 calculators (ABACUS datasets)
calculators.zip           calculator implementations (unzip to get calculators/)
data/enriched/
  abacus212_dataset.csv   ABACUS212 (212 samples, enriched)
  abacus409_dataset.csv   ABACUS409 (397 samples, enriched)
requirements.txt
```

Unzip `calculators.zip` in the repository root before running (it creates `calculators/`). MedCalc-Bench is not redistributed here: obtain it from its original source and place the CSV in `data/`; its 55-calculator MCP registry is needed only for `DATASET_NAME = 'medcalc'` in the MCP notebook.

Example configuration (first cell of each notebook):

```
DATASET_NAME      = 'abacus212'
DATASET_FILE      = 'data/enriched/abacus212_dataset.csv'
MCP_REGISTRY_FILE = 'registry/mcp_registry_abacus.json'
CALCULATORS_DIR   = 'calculators/'
```

## Setup

```
pip install -r requirements.txt
```

A CUDA GPU is required (the model is loaded in 4-bit NF4). Gated models need `HF_TOKEN` in the environment; no token is stored in the notebooks.

## Inputs (set in the Configuration cell of each notebook)

| Setting | Meaning |
|---|---|
| `DATASET_NAME` | label used in the output directory (MCP: `abacus212`, `abacus409` or `medcalc`) |
| `DATASET_FILE` | dataset CSV; the baselines map common column names automatically (`MANUAL_COLUMN_OVERRIDE` for unusual ones) and need these fields: calculator name, category, patient note, question, relevant entities, ground-truth answer, lower limit, upper limit. The MCP notebook expects the ABACUS columns (`Calculator`, `query` / `vignette`, `Category`, `ground_truth`, `relevant_entities`, `lower_limit`, `upper_limit`) for `abacus212` / `abacus409`, and the MedCalc-Bench columns (`Calculator Name`, `Category`, `Patient Note`, `Question`, `Relevant Entities`, `Ground Truth Answer`, `Lower Limit`, `Upper Limit`) for `medcalc` (55-calculator registry) |
| `MODEL_ID` | Hugging Face id of any chat model that fits the GPU in 4-bit |
| `CALCS_DIR` | `calculators/` folder (contains `name_to_python.json`; used by Zero-Shot Direct, Tool Calling, RAG + CI) |
| `MCP_REGISTRY_FILE`, `CALCULATORS_DIR` | MCP notebook only |

Results are written to `results/<strategy>_<dataset>_<model>/`.

## Reproducibility

Seed 42, greedy decoding (`do_sample=False`) and 4-bit NF4 quantization in every notebook. See Part A of the *Evaluation protocol* section (identical in all notebooks) and Part B (pipeline-specific denominators and rules).
