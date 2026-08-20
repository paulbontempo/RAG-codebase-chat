from dataclasses import dataclass
from pathlib import Path

import yaml

BENCHMARK_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BENCHMARK_DIR / "repo_manifest.yaml"


@dataclass
class BenchmarkRepo:
    name: str
    url: str
    ref: str
    commit: str
    package_root: str


def load_manifest(path: Path = MANIFEST_PATH) -> list[BenchmarkRepo]:
    data = yaml.safe_load(path.read_text())
    return [BenchmarkRepo(**repo) for repo in data["repos"]]
