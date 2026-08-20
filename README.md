# codebase-chat-tool

I've inherited a legacy codebase before. The original devs were gone, there was conflicting documentation, and a deadline was coming. This is the tool I wish I'd had in that situation: a CLI chatbot that lets you ask questions about an unfamiliar Python codebase *and* tells you exactly what will break before you apply your changes, all grounded in hybrid RAG retrieval.

> **Status**: functional end-to-end (indexing, hybrid retrieval, agentic chat, blast-radius analysis, deterministic eval harness). Demo recording is still pending.

## Demo

*pending*

## What it does

Two workflows on one index of your codebase:

- **`chat` / `ask`** — RAG-backed agentic Q&A over the entire codebase. The LLM decides when to search, fetch a definition, or query the call graph, and every factual claim is cited as `file_path:line`, so that you can trace references as needed.
- **`impact`** — analysis of potential downstream effects *before* you make changes. Give it a function or class, and it walks the *static call graph* (deterministic, no LLM hallucinations) to show every direct/transitive caller, every module that imports it, and every test that looks like it covers it. The LLM is used only to summarize the risk in plain English.

The blast-radius analysis is the part that differentiates this from a generic "chat with your repo" project: most RAG-over-code demos can answer *"what does this do?"*, but this one can also answer *"what will I break?"*, using deterministic static analysis (an `ast`-based call/import graph).

## What it looks like

Real, verbatim output against [`psf/requests`](https://github.com/psf/requests) (`v2.34.2`):

```
$ codebase-chat-tool impact requests.sessions.Session.send

Impact analysis: requests.sessions.Session.send

           Direct callers
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Caller                            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ requests.sessions.Session.request │
└───────────────────────────────────┘
    Additional transitive callers
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Caller                            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ requests.sessions.Session.delete  │
│ requests.sessions.Session.get     │
│ requests.sessions.Session.head    │
│ requests.sessions.Session.options │
│ requests.sessions.Session.patch   │
│ requests.sessions.Session.post    │
│ requests.sessions.Session.put     │
└───────────────────────────────────┘
 Related tests
┏━━━━━━━━━━━━━━┓
┃ Test         ┃
┡━━━━━━━━━━━━━━┩
│ (none found) │
└──────────────┘

Risk summary:
The blast radius for the proposed change to `requests.sessions.Session.send` is
significant, as it is directly called by `requests.sessions.Session.request` and
indirectly affects seven other HTTP methods: delete, get, head, options, patch,
post, and put. However, a major concern is that there are no tests associated
with this symbol, which leaves all callers unverified in terms of their
functionality when changes are made. Given this lack of coverage, the developer
should proceed with caution, as any alterations to the method's signature or
behavior could lead to widespread, untested failures across multiple modules.
```

("No tests found" here is a real, correctly-reported result. `requests`' test suite lives outside the `src/` layout this was indexed at. See [Limitations](#limitations) (not a false negative from the tool.))

## Architecture

```
Repo (local clone)
  -> ingestion/   tree-sitter structural chunking (function/class/module chunks)
  -> graph/       ast-based call graph + import graph (networkx)
  -> retrieval/   Qdrant dense + BM25, Reciprocal Rank Fusion, cross-encoder rerank
  -> agent/       LangGraph: chat_graph (agentic RAG) + impact_graph (deterministic + LLM synthesis)
  -> llm/         provider-agnostic: Anthropic or OpenAI, swappable via .env
  -> cli/         typer + rich
```

Two LangGraph workflows share that substrate:
- **`chat_graph`** — the LLM can call `search_code` / `get_definition` / `get_callers` / `get_callees` / `get_importers` / `find_tests_for` mid-conversation, rather than a hardcoded retrieve-then-answer pipeline.
- **`impact_graph`** — almost entirely deterministic graph traversal; the LLM is used *only* at the very end, to turn a list of callers/dependents/tests into a readable risk summary. I don't trust an LLM to enumerate call sites — I trust static analysis for that, and use the LLM only to prioritize/explain.

See [`docs/architecture.md`](docs/architecture.md) for more detail, including how the eval harness works.

## Evaluation

Every metric below is computed **deterministically** — exact set/string comparisons, no LLM-as-judge anywhere. Run against 3 pinned real-world repos (`requests`, `flask`, `typer`) with hand-verified gold answers in [`src/codebase_chat_tool/eval/benchmark/`](src/codebase_chat_tool/eval/benchmark/):

| Repo | P@k | R@k | MRR | Citation rate | Grounded rate | Keyword coverage | Caller P | Caller R | Caller F1 | Test F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| requests | 0.29 | 1.00 | 0.66 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| flask | 0.12 | 0.92 | 0.58 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| typer | 0.10 | 0.67 | 0.33 | 0.50 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 | 0.40 |

(Full breakdown per repo: [`docs/eval_report.md`](docs/eval_report.md). Run it yourself: `codebase-chat-tool eval`.)

- **P@k / R@k / MRR** — standard IR metrics for the retriever, against hand-labeled gold qualnames.
- **Citation / grounded rate** — how often the agent cites a source, and of those citations, what fraction point at a location it *actually saw* via a tool call (mechanically verified, not LLM-judged).
- **Caller P/R/F1, Test F1** — blast-radius accuracy against hand-verified gold caller/test sets, computed directly from the static call graph with zero LLM calls.

Low P@k is expected and not a bug: `top_k` defaults to 8, but most gold answers are 1-3 symbols, so precision is mechanically capped well below 1.0 even for a perfect retriever. Recall is the metric that matters here, and it's consistently strong.

CI runs this on every PR touching `src/codebase_chat_tool/**` and fails the build if any metric regresses more than 0.10 versus the committed baseline (`docs/eval_baseline.json`) — see [`.github/workflows/eval.yml`](.github/workflows/eval.yml).

## Quickstart

```bash
git clone <this-repo>
cd codebase-chat-tool
docker-compose up -d          # starts Qdrant
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # fill in ANTHROPIC_API_KEY or OPENAI_API_KEY, set LLM_PROVIDER

codebase-chat-tool index /path/to/some/repo
codebase-chat-tool chat --repo /path/to/some/repo
codebase-chat-tool impact some_module.SomeClass.some_method --repo /path/to/some/repo
```

No API key yet? `codebase-chat-tool index` and `codebase-chat-tool graph callers <symbol>` both work with zero LLM calls — the call-graph analysis is pure static analysis.

## Design decisions & tradeoffs

- **tree-sitter for chunking, `ast` for the call graph — not one tool for both.** tree-sitter is error-tolerant (parses a file even with syntax errors elsewhere) and gives precise byte/line spans, which is what chunk boundaries need. Python's `ast` gives fully-typed, semantically rich nodes with zero extra dependency, which is what call-graph construction needs. Each tool is used for what it's actually good at.
- **Qdrant over Chroma/pgvector.** Payload filtering, HNSW internals, and a real production deployment story (Qdrant Cloud) — reads as production-signal rather than a notebook dependency, and is the more common interview topic for this role family.
- **Reciprocal Rank Fusion + cross-encoder rerank, not dense-only retrieval.** Code search is a case where identifier/keyword matching (BM25) frequently beats pure embedding similarity — hybrid retrieval with RRF fusion is the standard fix, cheap to implement, and easy to defend in an interview.
- **A hand-rolled provider-agnostic LLM layer, not a direct SDK dependency in business logic.** Anthropic and OpenAI have genuinely different tool-calling wire formats; normalizing both into one `LLMProvider` interface is a real, demonstrable abstraction problem, and it means agent/business logic never imports a vendor SDK.
- **Fully deterministic eval, not RAGAS.** I originally planned to use RAGAS for QA evaluation. Its latest release has a broken import (`langchain_community.chat_models.vertexai`, which no longer exists), and the last version that imports cleanly requires `langchain-core<0.3`, incompatible with the `langgraph>=1.2` this project runs on. Rather than downgrade the whole agent stack to a stale LangChain generation, I built a fully deterministic eval suite instead — standard IR metrics, mechanical citation verification, keyword coverage, and set-based blast-radius accuracy. RAGAS's own faithfulness/relevancy metrics are LLM-judge based internally anyway, so this isn't a downgrade in rigor — it removes the "LLM grading an LLM" black box entirely and makes every number in the report auditable by reading the eval code.

## Limitations

- **Python only, local clones only.** No multi-language support, no `git clone` of a remote URL for you.
- **Best-effort call resolution.** Dynamic dispatch (`getattr`, calls through a variable whose type isn't known statically) is correctly reported as *unresolved*. you'll see this in `graph callers` output as "no resolved callers found" even where a real (polymorphic) caller exists.
- **`self`/`cls` resolution doesn't cross class inheritance.** A call to `self.method()` resolves within the *defining* class, not up the MRO, so an abstract method overridden in a subclass may show as called from the wrong class.
- **Decorator expressions aren't call-graph-tracked.** `@app.route(...)` registers a route via a decorator; the decorator's own call isn't currently walked for the call graph (only calls inside function/method bodies are).
- **`find_tests_for` is a heuristic** (short-name substring match in files under `tests/`), not a true call-graph link into test files. Therefore, it will occasionally over- or under-match. Future versions could add embedding similarity search to improve matching. 

## Roadmap

- Web UI (the core library/agent layer is already decoupled from the CLI, so this is additive, not a rewrite)
- Multi-language support (tree-sitter has grammars for most languages already; the `ast`-based call graph would need a per-language equivalent)
- Arbitrary GitHub URL ingestion (currently local-clone-only by design, to keep v1 scope tight)

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```
