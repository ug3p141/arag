# ARAG - Agentic Retrieval-Augmented Generation Framework

## Project Overview

Research framework implementing hierarchical retrieval interfaces for agentic RAG. An LLM agent uses three retrieval tools (`keyword_search`, `semantic_search`, `read_chunk`) in a ReAct loop to answer multi-hop questions over document corpora.

Paper: https://arxiv.org/abs/2602.03442

## Quick Reference

- **Language**: Python 3.10+
- **Package manager**: uv
- **Package**: `src/arag/` (hatchling build, src layout)
- **Config files**: YAML in `configs/`
- **Environment variables**: `ARAG_API_KEY`, `ARAG_BASE_URL`, `ARAG_MODEL`

## Commands

```bash
# Install dependencies
uv sync --extra full      # full (with embeddings)
uv sync --extra dev       # dev (with pytest/ruff/mypy)

# Run tests
uv run pytest tests/

# Lint and format
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/

# Type check
uv run mypy src/arag/

# Build embedding index
uv run python scripts/build_index.py --chunks data/<dataset>/chunks.json --output data/<dataset>/index --model Qwen/Qwen3-Embedding-0.6B --device cuda:0

# Run batch evaluation
uv run python scripts/batch_runner.py --config configs/example.yaml --questions data/<dataset>/questions.json --output results/<dataset> --workers 5

# Evaluate results
uv run python scripts/eval.py --predictions results/<dataset>/predictions.jsonl --workers 5
```

## Architecture

```
src/arag/
├── core/
│   ├── config.py      # Config: loads YAML/JSON, dot-notation access
│   ├── context.py     # AgentContext: tracks read chunks, token budgets, retrieval logs
│   └── llm.py         # LLMClient: OpenAI-compatible API client with cost tracking
├── agent/
│   ├── base.py        # BaseAgent: ReAct loop with tool calling, token budget, max loops
│   └── prompts/       # System prompt templates
└── tools/
    ├── base.py        # BaseTool ABC: name, get_schema(), execute(context, **kwargs) -> (str, dict)
    ├── registry.py    # ToolRegistry: register/execute tools, schema collection
    ├── keyword_search.py   # Exact lexical matching, no index needed
    ├── semantic_search.py  # Dense retrieval with sentence embeddings, requires index
    └── read_chunk.py       # Full chunk retrieval by ID, supports ±1 adjacent context
```

## Key Patterns

- **Tool interface**: Subclass `BaseTool`, implement `name`, `get_schema()` (OpenAI function calling format), `execute(context, **kwargs) -> Tuple[str, Dict]`
- **Agent loop**: `BaseAgent.run(query)` returns `{"answer", "trajectory", "total_cost", "loops", ...}`
- **Context tracking**: `AgentContext` prevents redundant chunk reads via `mark_chunk_as_read()`/`is_chunk_read()`
- **LLM client**: Uses raw `requests` against OpenAI-compatible `/chat/completions` endpoint; includes cost calculation from token usage
- **Config**: `Config.from_yaml(path)` with dot-notation access (`config.get("llm.model")`)

## Code Style

- Formatter/linter: ruff (line-length 100, target py310)
- Type hints on all public method signatures
- Docstrings on classes and public methods
- Tests in `tests/` using pytest (no fixtures file yet, tests are self-contained)

## Do Not

- Edit `.env` files or `uv.lock`
- Hardcode API keys; use `ARAG_*` environment variables
- Add dependencies without discussing first; the project intentionally has a minimal core dependency set
- Break the `BaseTool` interface contract (`name`, `get_schema`, `execute`)
