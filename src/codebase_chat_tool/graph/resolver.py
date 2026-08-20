from dataclasses import dataclass

import networkx as nx


@dataclass
class CallerInfo:
    qualname: str
    lineno: int
    resolved: bool


class GraphResolver:
    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def _call_edges_to(self, qualname: str) -> list[CallerInfo]:
        callers: list[CallerInfo] = []
        for u, _v, data in self.graph.in_edges(qualname, data=True):
            if data.get("type") == "call":
                callers.append(
                    CallerInfo(
                        qualname=u,
                        lineno=data.get("lineno", 0),
                        resolved=data.get("resolved", False),
                    )
                )
        return callers

    def direct_callers(self, qualname: str) -> list[CallerInfo]:
        return [c for c in self._call_edges_to(qualname) if c.resolved]

    def transitive_callers(self, qualname: str, max_depth: int = 5) -> list[CallerInfo]:
        seen: dict[str, CallerInfo] = {}
        frontier = [qualname]
        depth = 0
        while frontier and depth < max_depth:
            next_frontier: list[str] = []
            for node in frontier:
                for caller in self.direct_callers(node):
                    if caller.qualname not in seen and caller.qualname != qualname:
                        seen[caller.qualname] = caller
                        next_frontier.append(caller.qualname)
            frontier = next_frontier
            depth += 1
        return list(seen.values())

    def direct_callees(self, qualname: str) -> list[CallerInfo]:
        callees: list[CallerInfo] = []
        for _u, v, data in self.graph.out_edges(qualname, data=True):
            if data.get("type") == "call" and data.get("resolved"):
                callees.append(CallerInfo(qualname=v, lineno=data.get("lineno", 0), resolved=True))
        return callees

    def importers(self, module: str) -> list[str]:
        result = []
        for u, _v, data in self.graph.in_edges(module, data=True):
            if data.get("type") == "import":
                result.append(u)
        return result

    def unresolved_calls_from(self, qualname: str) -> list[str]:
        return [
            v
            for _u, v, data in self.graph.out_edges(qualname, data=True)
            if data.get("type") == "call" and not data.get("resolved")
        ]
