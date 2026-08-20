from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from codebase_chat_tool.config import get_settings
from codebase_chat_tool.graph.resolver import GraphResolver
from codebase_chat_tool.graph.store import load_graph

graph_app = typer.Typer(help="Debug/inspect the indexed call and import graph directly (no LLM).")
console = Console()


def _load_resolver(repo: str) -> GraphResolver:
    settings = get_settings()
    repo_root = Path(repo).resolve()
    graph_path = settings.index_path(repo_root) / "graph.pkl"
    if not graph_path.exists():
        console.print(
            f"[red]No index found at {graph_path}.[/red]\n"
            f"[red]Run `codebase-chat-tool index {repo}` first.[/red]"
        )
        raise typer.Exit(1)
    return GraphResolver(load_graph(graph_path))


@graph_app.command()
def callers(
    symbol: str = typer.Argument(
        ..., help="Fully-qualified symbol, e.g. 'service.UserService.get_user'."
    ),
    repo: str = typer.Option(".", "--repo", help="Path to the indexed repository."),
    transitive: bool = typer.Option(
        False, "--transitive", help="Include transitive (indirect) callers."
    ),
) -> None:
    """List callers of SYMBOL."""
    resolver = _load_resolver(repo)
    results = resolver.transitive_callers(symbol) if transitive else resolver.direct_callers(symbol)
    if not results:
        console.print(f"[yellow]No resolved callers found for {symbol}[/yellow]")
        return
    table = Table(title=f"Callers of {symbol}")
    table.add_column("Caller")
    table.add_column("Line")
    for c in results:
        table.add_row(c.qualname, str(c.lineno))
    console.print(table)


@graph_app.command()
def callees(
    symbol: str = typer.Argument(
        ..., help="Fully-qualified symbol, e.g. 'service.UserService.create_user'."
    ),
    repo: str = typer.Option(".", "--repo", help="Path to the indexed repository."),
) -> None:
    """List symbols that SYMBOL calls."""
    resolver = _load_resolver(repo)
    results = resolver.direct_callees(symbol)
    if not results:
        console.print(f"[yellow]No resolved callees found for {symbol}[/yellow]")
        return
    table = Table(title=f"Callees of {symbol}")
    table.add_column("Callee")
    table.add_column("Line")
    for c in results:
        table.add_row(c.qualname, str(c.lineno))
    console.print(table)


@graph_app.command()
def importers(
    module: str = typer.Argument(..., help="Module name, e.g. 'utils'."),
    repo: str = typer.Option(".", "--repo", help="Path to the indexed repository."),
) -> None:
    """List modules that import MODULE."""
    resolver = _load_resolver(repo)
    results = resolver.importers(module)
    if not results:
        console.print(f"[yellow]No importers found for {module}[/yellow]")
        return
    for m in results:
        console.print(m)
