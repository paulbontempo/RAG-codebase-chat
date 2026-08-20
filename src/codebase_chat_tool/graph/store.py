import pickle
from pathlib import Path

import networkx as nx


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(graph, f)


def load_graph(path: Path) -> nx.MultiDiGraph:
    with path.open("rb") as f:
        return pickle.load(f)
