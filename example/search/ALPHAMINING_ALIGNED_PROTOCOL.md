# AlphaBench-EA × AlphaMining aligned protocol

This workflow runs directly on Python 3.11.8. It does not use Docker, Conda,
Docker-in-Docker, or an image build.

## Fixed experiment contract

- Qlib `cn_data`; fields: `open/high/low/close/volume/vwap`
- PIT universe `csi300`; benchmark `SH000300`
- train: `2010-01-01` through `2019-11-30`
- validation: `2020-01-01` through `2021-11-30`
- held-out test: `2022-01-01` through `2025-12-31`
- label: `Ref($open, -11) / Ref($open, -1) - 1`
- comparison ruler: Qlib `LinearModel`, ridge, `alpha=10.0`, no intercept
- strategy: Qlib TopkDropout 50/5, open execution, account ¥100m,
  buy/sell costs 5/15 bps, minimum order cost ¥5

Here “10% turnover cap” means at most `5/50=10%` of names are replaced per
day. Qlib's report column is two-sided traded value (sell plus buy), so it can
be near `2×5/50=20%`; that is the same TopkDropout protocol, not a mismatch.

The December 2019 and December 2021 gaps are deliberate embargoes. Qlib can
evaluate negative `Ref` beyond a requested segment boundary, so those gaps keep
the final 10-day label of train/validation away from the next segment.

## Server setup (no Docker)

Run from the AlphaBench repository root:

```bash
pyenv shell 3.11.8
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[research]'

export QLIB_PROVIDER_URI="$HOME/.qlib/qlib_data/cn_data"
export QLIB_DATA_PATH="$QLIB_PROVIDER_URI"
export DEEPSEEK_API_KEY="..."

python searcher/validate_protocol.py
ppo start backend --workers 1
ppo status
```

The validator reads only tiny train/validation samples. It does not read test
returns.

## Run EA search

```bash
python searcher/start_search.py \
  --config example/search/configs/alphabench_ea_alphamining.yaml
```

The run produces:

```text
runs/T3/alphabench_ea_alphamining_csi300/
├── final_pool.jsonl
├── factor_library/
│   ├── registry.json       # AlphaMining-compatible (`id` + `qlib_expr`)
│   ├── manifest.json       # complete protocol, test_evaluated=false
│   └── factors.csv         # easy train/validation inspection
└── backtest_records/       # train/validation evidence only
```

AlphaMining can consume the exported registry directly:

```bash
cd ../alpha_mining
.venv/bin/python scripts/run_backtest.py \
  --registry ../AlphaBench/runs/T3/alphabench_ea_alphamining_csi300/factor_library/registry.json \
  --segment valid
```

## Side-by-side public comparison

Validation is safe to run repeatedly:

```bash
python searcher/compare_factor_libraries.py --segment valid
```

The held-out test requires an explicit public-evaluation flag and is never
called by search:

```bash
python searcher/compare_factor_libraries.py --segment test --public-test
```

Outputs are written under `runs/library_comparison/<segment>/` as JSON, CSV,
Markdown, a side-by-side PNG, cumulative-return CSV, and an auditable
`protocol.json`. The
comparison changes only the factor registry; target, Ridge ruler, processors,
universe, execution, costs, and portfolio strategy stay fixed.

For CSI500 zero-shot, copy the YAML and change `market` to `csi500` plus the
benchmark to `SH000905`. Keep the original CSI300 run immutable. SPY/SP500
requires a compatible US Qlib bundle and should be treated as a separate
cross-market protocol rather than silently reusing CN costs and price limits.
