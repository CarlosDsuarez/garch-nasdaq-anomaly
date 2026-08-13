# AGENTS.md

## Cursor Cloud specific instructions

This is a single-file Python data-science app (`main.py`): the **GARCH NASDAQ
Anomaly Detector**. It downloads QQQ price history from Yahoo Finance via
`yfinance`, fits a GARCH(1,1) model, computes a rolling Z-score, classifies
volatility regimes, and writes plots + a CSV to `outputs/` (and cached returns
to `data/`).

### Environment / running

- Python dependencies live in a virtualenv at `.venv/` (created during setup;
  the update script keeps it in sync with `requirements.txt`). Activate with
  `. .venv/bin/activate`, then run the app with `python main.py`.
- There is **no lint config and no test suite** in this repo. The closest
  "lint" check is `python -m py_compile main.py` (syntax check). Do not add lint
  or test infrastructure unless asked.
- Running `python main.py` overwrites the tracked files `data/qqq_returns.csv`
  and `outputs/*` with freshly downloaded data. These are runtime byproducts —
  do **not** commit the regenerated versions; `git checkout -- data outputs`
  to discard them before committing unrelated changes.

### Yahoo Finance User-Agent gotcha (important)

`yfinance==0.2.51` (pinned in `requirements.txt`) ships a hardcoded, years-old
`Chrome/39` User-Agent that Yahoo now rate-limits with HTTP 429
("Edge: Too Many Requests") on the crumb endpoint. With the stock UA every data
download fails with `possibly delisted; no price data found` even though the
network and Yahoo are reachable.

The fix is an **environment-level shim**, not a code change: the venv's
site-packages contains `_yf_ua_patch.py` + `zzz_yf_ua_patch.pth`, which patch
`yfinance.data.YfData.user_agent_headers` to a modern Chrome UA at interpreter
startup. A `.pth` file is used (not `sitecustomize.py`) because the base image
ships its own `sitecustomize.py` that shadows a venv-local one. pip does not
manage these files, so the shim survives `pip install -r requirements.txt`.
If you ever recreate `.venv` from scratch, re-add both files or the app's data
downloads will start failing with 429s again.

If a run fails with a stale-crumb error, clear the yfinance cache with
`rm -rf ~/.cache/py-yfinance` and retry.
