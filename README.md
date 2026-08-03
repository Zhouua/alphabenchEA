# AlphaBench-EA

Minimal, no-Docker workflow for evolutionary alpha-factor discovery and an
AlphaMining-aligned public evaluation. The repository intentionally contains
only the EA search path, its Qlib factor oracle, Alpha158 seeds, export tools,
and the fixed comparison protocol.

## Fixed protocol

- Qlib `cn_data`: `open`, `high`, `low`, `close`, `volume`, `vwap`
- CSI300 universe; `SH000300` benchmark
- train: `2010-01-01` to `2019-11-30`
- validation: `2020-01-01` to `2021-11-30`
- held-out test: `2022-01-01` to `2025-12-31`
- target: `Ref($open, -11) / Ref($open, -1) - 1`
- ruler: Qlib LinearModel Ridge, `alpha=10`, `fit_intercept=False`
- TopK 50 / Drop 5, open execution, CNY 100m account, 5/15 bps costs

The test segment is unavailable to search and validation. It can only be run
through the explicit public-evaluation flag.

## Server setup

Python 3.11.8 is required. Docker and Conda are not used.

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
python searcher/start_search.py
```

The EA run exports an AlphaMining-compatible registry to:

```text
runs/T3/alphabench_ea_alphamining_csi300/factor_library/registry.json
```

Compare the AlphaBench-EA and AlphaMining libraries on validation:

```bash
python searcher/compare_factor_libraries.py --segment valid
```

Run the held-out public test only once the library is frozen:

```bash
python searcher/compare_factor_libraries.py --segment test --public-test
```

See [the full aligned protocol](example/search/ALPHAMINING_ALIGNED_PROTOCOL.md)
for output files, AlphaMining registry interoperability, and zero-shot notes.

## Runtime layout

```text
agent/          LLM generation and Qlib expression validation
factors/        Alpha158 seed factors required by EA
ffo/            factor evaluation backend and client
searcher/       EA pipeline, export, validation, and library comparison
example/search/ the single fixed experiment configuration and protocol
config/         LLM provider routing (environment-variable credentials)
```
