# Setup Guide

## 1. Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | 3.12+ | Modern typing syntax; `uv` manages the interpreter |
| [uv](https://docs.astral.sh/uv/) | latest | Resolver + lockfile + venv + runner in one tool |
| Docker | any recent | Runs Qdrant locally |
| Git | any | — |

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Install the project

```bash
uv sync --all-groups
```

What this does, and why it is not `pip install -r requirements.txt`:

- Resolves from `pyproject.toml`, writes/uses `uv.lock` — **exact** versions,
  reproducible on every machine and in CI
- Creates `.venv/` automatically
- Installs the project itself in editable mode, so `import secfiler_rag` works
  from the installed package (this is what makes the `src/` layout safe)

You never need to activate the venv: `uv run <cmd>` runs inside it and re-syncs
if the lockfile changed.

## 3. Configure the environment

```bash
cp .env.example .env
```

Then fill in what the module you are working on needs:

| Variable | Needed from | Notes |
|---|---|---|
| `ENVIRONMENT` | now | `local` / `ci` / `staging` / `production` |
| `LOG_LEVEL` | now | `DEBUG` while developing |
| `LOG_FORMAT` | now | `console` locally, `json` in deployment |
| `DATA_DIR` | now | Defaults to `data` |
| `OPENAI_API_KEY` | Module 2 (indexing) | [platform.openai.com](https://platform.openai.com) |
| `QDRANT_URL` | Module 2 | `http://localhost:6333` for local Docker |
| `QDRANT_API_KEY` | Module 2 | Blank for local |
| `LANGSMITH_TRACING` | Module 7 | `false` by default — the system runs without LangSmith |
| `LANGSMITH_API_KEY` | Module 7 | Only if tracing is on |

`.env` is git-ignored. `.env.example` is committed and must list every
variable — an undocumented env var is a production incident waiting to happen.

Unknown keys in `.env` are ignored by `Settings` (`extra="ignore"`), so tools
that share the file cannot crash the app.

## 4. Start Qdrant

```bash
docker compose up -d
```

Verify:

```bash
curl -s http://localhost:6333/healthz
```

Dashboard: <http://localhost:6333/dashboard>

Storage persists in `qdrant_storage/` (git-ignored). To start from a clean
vector store, stop the container and delete that directory — everything in it
is rebuildable by re-running indexing.

## 5. Verify the install

```bash
uv run pytest -m "not integration"
```

```bash
uv run ruff check src tests
```

```bash
uv run mypy
```

All three should pass on a fresh clone. If they do not, that is a bug in the
repo, not in your machine — please treat it that way.

## 6. Common setup problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: secfiler_rag` | Ran `python` directly instead of `uv run` | `uv run python …` |
| `ValidationError` on startup naming a field | Env var missing or misspelled | Compare `.env` against `.env.example` |
| Connection refused on `:6333` | Qdrant not running | `docker compose up -d` |
| Tests pass locally, fail in CI | A test read your real `.env` | Use the `clean_env` fixture / `IsolatedSettings` |
| `uv sync` resolution error | Stale lock after editing deps | `uv lock --upgrade` then `uv sync` |
