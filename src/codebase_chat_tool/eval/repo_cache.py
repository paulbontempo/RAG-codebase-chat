import subprocess
from pathlib import Path

from codebase_chat_tool.eval.benchmark.manifest import BenchmarkRepo

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / ".eval_cache" / "repos"


def ensure_benchmark_repo(repo: BenchmarkRepo, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Clones the repo into cache_dir if needed, and checks out the pinned commit.
    Idempotent: safe to call repeatedly."""
    dest = cache_dir / repo.name
    if not dest.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", repo.url, str(dest)], check=True)

    current = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if current != repo.commit:
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--quiet", "origin", repo.commit], check=False
        )
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", repo.commit], check=True)

    return dest / repo.package_root
