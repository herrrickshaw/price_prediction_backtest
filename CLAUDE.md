# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## Layout and commands
- Package `price_prediction/`: `data.py` (parquet I/O, universe), `baselines.py`, `markov_regime.py`,
  `kalman_trend.py`, `features.py` (torch-free), `lstm_model.py` (optional torch), `evaluate.py`
  (walk-forward harness), `replay.py` (regime-gated Darvas replay), `__main__.py` (CLI).
- Run: `python -m price_prediction --market US --top 20 --test-days 500`;
  `python -m price_prediction.replay --market US IN JP KR CN`. Panels come from `$LTM_DIR`.
- Verify: `ruff check .` and `pytest -q` (synthetic seeded series only; LSTM tests skip without torch).

## Rules
- The Darvas box in `replay.darvas_signals` is built with `.shift(1)`: the current bar is always
  excluded from its own box. Never change that.
- Never download market data in tests; never commit panels. `reports/` and `graphify-out/` are
  generated artefacts — do not hand-edit.
- Behaviour-preserving refactors only; `tests/fixtures/baseline.json` pins the pre-refactor outputs.

<!-- source-fetch-toolkit:v1 (managed by ~/bin/add_source_fetch_section.py — edit there, not here) -->
## Source fetching for data collection & literature surveys

When a task needs a primary document (statute, regulator data, government release, paper, filing),
fetch the document itself — never cite a search snippet as if it had been read.

- **On the owner's Mac:** `fetch_source.py "<url | doi:… | celex:… | arxiv:…>" --out ./sources`
  (in `~/bin`; full guide `~/bin/SOURCE_FETCH_TOOLKIT.md`). Scheduled/unattended runs use only
  `bash /Users/umashankar/bin/fetch_source.sh "<target>"` (allowlisted) and then Read the `.txt` it reports.
- **Elsewhere (cloud sessions, CI):** follow the same ladder by hand — direct GET with a browser User-Agent →
  headless browser only if the page needs JavaScript → official alternates (DOI → OpenAlex
  `api.openalex.org/works/doi:<doi>` locations → repository copy; EUR-Lex → EU Publications Office
  `publications.europa.eu/resource/celex/<CELEX>`; arXiv pdf/abs) → Wayback Machine `id_` snapshot.
- **Hard rule:** a CAPTCHA, robot check, Cloudflare/Akamai/WAF challenge or login wall means stop on that
  host. Do not bypass it (no TLS impersonation, stealth browsers, identity rotation or challenge solving);
  look for the same document via another route, or report it as unavailable.
- **Record** for every figure: URL read, title/date, page/table, fetch route; mark abstract-only or
  snippet-only sources. Fetch politely (~1 request/second per host, no bulk crawling).
<!-- /source-fetch-toolkit -->
