# Architecture

See the project plan for full context. Summary:

```
Repo (local clone)
  -> ingestion/ (tree-sitter structural chunking)
  -> graph/ (ast-based call graph + import graph, networkx)
  -> retrieval/ (Qdrant dense + BM25, RRF fusion, cross-encoder rerank)
  -> agent/ (LangGraph: chat_graph, impact_graph)
  -> llm/ (provider-agnostic: Anthropic / OpenAI)
  -> cli/ (typer + rich)
```

Two workflows:
- **chat_graph**: agentic RAG — the LLM can call retrieval/graph tools mid-conversation, cites file:line.
- **impact_graph**: deterministic call-graph/import-graph traversal (no LLM) to find callers, dependents, and tests for a proposed change; LLM only synthesizes the final risk summary.

## Evaluation

The eval harness (`eval/`) runs against three pinned real-world repos (`requests`, `flask`, `typer`; see `eval/benchmark/repo_manifest.yaml` for exact commits) with hand-verified gold answers (`eval/benchmark/questions_*.jsonl`). Every metric is computed by exact, deterministic comparison — there is no LLM-as-judge step anywhere in this report, by design:

- **Retrieval — precision@k, recall@k, MRR** (`eval/retrieval_metrics.py`): standard IR metrics comparing the retriever's ranked output against hand-labeled gold qualnames.
- **Groundedness — citation rate / grounded rate** (`eval/groundedness.py`): every `file_path:line` citation the chat agent produces is mechanically checked against the tool-call results it actually received during that conversation — did it cite something it was really shown, rather than trusting an LLM's self-report of faithfulness.
- **Correctness — keyword coverage** (`eval/correctness.py`): exact case-insensitive substring match of hand-picked required facts against the answer.
- **Blast radius — caller/test precision, recall, F1** (`eval/impact_eval.py`): set comparison between the static call graph's direct callers / test matches and hand-verified gold sets. No LLM call is made for this metric at all.

We initially planned to use RAGAS for the QA metrics, but its latest release (0.4.3) has a broken import (`langchain_community.chat_models.vertexai`, which no longer exists) and the last version that imports cleanly (0.1.21) requires `langchain-core<0.3`, incompatible with the `langgraph>=1.2` this project is built on. RAGAS's faithfulness/relevancy metrics are themselves LLM-judge based internally, so a hand-rolled LLM-judge wouldn't have been a meaningfully different (or more standardized) evaluation anyway. Fully deterministic, mechanically-checked metrics avoid that "LLM grading an LLM" problem entirely and make every number in the report auditable by reading the eval code.
