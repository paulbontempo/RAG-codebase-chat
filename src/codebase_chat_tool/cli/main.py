from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from codebase_chat_tool import __version__
from codebase_chat_tool.cli.graph_app import graph_app

app = typer.Typer(
    name="codebase-chat-tool",
    help="Understand and safely modify an inherited Python codebase.",
    no_args_is_help=True,
)
app.add_typer(graph_app, name="graph")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"codebase-chat-tool {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    pass


@app.command()
def index(
    path: str = typer.Argument(..., help="Path to a locally cloned Python repository."),
) -> None:
    """Index a repository: chunk source files and build the call/import graph."""
    from codebase_chat_tool.ingestion.pipeline import run_indexing

    run_indexing(path)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language or keyword search query."),
    repo: str = typer.Option(".", "--repo", help="Path to the indexed repository."),
    top_k: int = typer.Option(8, "--top-k", help="Number of results to return."),
) -> None:
    """Search indexed code with hybrid retrieval (debug/inspection command)."""
    from codebase_chat_tool.config import get_settings
    from codebase_chat_tool.retrieval.retriever import Retriever

    settings = get_settings()
    repo_root = Path(repo).resolve()
    index_dir = settings.index_path(repo_root)
    if not (index_dir / "chunks.json").exists():
        console.print(
            f"[red]No index found at {index_dir}.[/red]\n"
            f"[red]Run `codebase-chat-tool index {repo}` first.[/red]"
        )
        raise typer.Exit(1)

    retriever = Retriever(repo_root, settings)
    results = retriever.search(query, top_k=top_k)
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f'Search results for "{query}"')
    table.add_column("Chunk")
    table.add_column("Dense rank")
    table.add_column("BM25 rank")
    table.add_column("Fused")
    table.add_column("Rerank")
    for r in results:
        table.add_row(
            f"{r.chunk.file_path}:{r.chunk.start_line} ({r.chunk.qualname})",
            str(r.dense_rank) if r.dense_rank else "-",
            str(r.bm25_rank) if r.bm25_rank else "-",
            f"{r.fused_score:.4f}",
            f"{r.rerank_score:.4f}" if r.rerank_score is not None else "-",
        )
    console.print(table)


def _load_ctx_and_provider(repo: str):
    from codebase_chat_tool.agent.context import load_repo_context
    from codebase_chat_tool.config import get_settings
    from codebase_chat_tool.llm.factory import get_provider

    settings = get_settings()
    repo_root = Path(repo).resolve()
    try:
        ctx = load_repo_context(repo_root, settings)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        provider = get_provider(settings)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    return ctx, provider


@app.command()
def ask(
    question: str = typer.Argument(..., help="A single question about the indexed repo."),
    repo: str = typer.Option(".", "--repo", help="Path to the indexed repository."),
) -> None:
    """Ask a single question about the indexed repository."""
    from codebase_chat_tool.agent.chat_graph import run_chat_turn

    ctx, provider = _load_ctx_and_provider(repo)
    with console.status("[cyan]Thinking...[/cyan]"):
        history = run_chat_turn(ctx, provider, [], question)
    console.print(history[-1].content)


@app.command()
def chat(
    repo: str = typer.Option(".", "--repo", help="Path to the indexed repository."),
) -> None:
    """Start an interactive chat session about the indexed repository."""
    from codebase_chat_tool.agent.chat_graph import run_chat_turn

    ctx, provider = _load_ctx_and_provider(repo)
    console.print("[green]codebase-chat-tool chat[/green] (type 'exit' or Ctrl-D to quit)")
    history: list = []
    while True:
        try:
            user_input = console.input("[bold cyan]you>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            continue
        with console.status("[cyan]Thinking...[/cyan]"):
            history = run_chat_turn(ctx, provider, history, user_input)
        console.print(f"[bold magenta]assistant>[/bold magenta] {history[-1].content}")


@app.command()
def impact(
    target: str = typer.Argument(
        ..., help="Symbol to analyze, e.g. 'pkg.module.function_or_class'."
    ),
    repo: str = typer.Option(".", "--repo", help="Path to the indexed repository."),
) -> None:
    """Show the blast radius (callers, dependents, tests) of changing TARGET."""
    from codebase_chat_tool.agent.impact_graph import run_impact_analysis

    ctx, provider = _load_ctx_and_provider(repo)
    with console.status("[cyan]Analyzing blast radius...[/cyan]"):
        result = run_impact_analysis(ctx, provider, target)

    if result.get("error"):
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Impact analysis: {result['resolved_target']}[/bold]\n")

    table = Table(title="Direct callers")
    table.add_column("Caller")
    for c in result.get("direct_callers", []) or ["(none)"]:
        table.add_row(c)
    console.print(table)

    if result.get("transitive_callers"):
        table = Table(title="Additional transitive callers")
        table.add_column("Caller")
        for c in result["transitive_callers"]:
            table.add_row(c)
        console.print(table)

    table = Table(title=f"Modules importing {result.get('target_module', '?')}")
    table.add_column("Module")
    for m in result.get("dependent_modules", []) or ["(none)"]:
        table.add_row(m)
    console.print(table)

    table = Table(title="Related tests")
    table.add_column("Test")
    tests = result.get("related_tests", [])
    for t in tests or [{"qualname": "(none found)"}]:
        table.add_row(t["qualname"])
    console.print(table)

    console.print("\n[bold]Risk summary:[/bold]")
    console.print(result.get("risk_summary") or "(unavailable)")


_REPOS_OPTION = typer.Option(
    None, "--repos", help="Benchmark repo names to run (default: all in the manifest)."
)


@app.command(name="eval")
def eval_cmd(
    repos: list[str] = _REPOS_OPTION,
    report_path: str = typer.Option("docs/eval_report.md", "--report-path"),
    results_path: str = typer.Option("eval_results.json", "--results-path"),
    baseline_path: str = typer.Option(
        "docs/eval_baseline.json",
        "--baseline-path",
        help="Committed baseline to check for regression against.",
    ),
    no_check_regression: bool = typer.Option(
        False, "--no-check-regression", help="Skip comparing against the baseline entirely."
    ),
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="Overwrite the baseline with this run's results."
    ),
) -> None:
    """Run the deterministic evaluation harness against the pinned benchmark repos.

    Exits non-zero if any checked metric regresses more than the tolerance
    versus the committed baseline (docs/eval_baseline.json) -- use
    --update-baseline to intentionally accept a new baseline.
    """
    from pathlib import Path as _Path

    from codebase_chat_tool.eval.run_eval import run_eval

    run_eval(
        repo_names=repos or None,
        report_path=_Path(report_path),
        results_path=_Path(results_path),
        baseline_path=None if no_check_regression else _Path(baseline_path),
        update_baseline=update_baseline,
    )


if __name__ == "__main__":
    app()
