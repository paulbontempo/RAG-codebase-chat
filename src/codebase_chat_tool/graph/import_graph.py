from codebase_chat_tool.graph.ast_visitor import FileAnalysis


def import_edges(analyses: list[FileAnalysis]) -> set[tuple[str, str]]:
    """Module -> imported-module edges, deduplicated."""
    edges: set[tuple[str, str]] = set()
    for fa in analyses:
        for imp in fa.imports:
            if imp.imported_module:
                edges.add((fa.module, imp.imported_module))
    return edges
