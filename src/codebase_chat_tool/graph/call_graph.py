import networkx as nx

from codebase_chat_tool.graph.ast_visitor import FileAnalysis
from codebase_chat_tool.graph.import_graph import import_edges


def build_graph(analyses: list[FileAnalysis]) -> nx.MultiDiGraph:
    """Builds one MultiDiGraph with both call edges (edge type='call') and
    import edges (edge type='import'). Def nodes carry kind/module node attrs."""
    graph = nx.MultiDiGraph()

    for fa in analyses:
        graph.add_node(fa.module, kind="module")
        for d in fa.defs:
            graph.add_node(d.qualname, kind=d.kind, module=fa.module, class_name=d.class_name)

    for fa in analyses:
        for call in fa.calls:
            graph.add_edge(
                call.caller_qualname,
                call.callee,
                type="call",
                lineno=call.lineno,
                resolved=call.resolved,
            )

    for src, dst in import_edges(analyses):
        graph.add_edge(src, dst, type="import")

    return graph
