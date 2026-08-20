from pathlib import Path

from codebase_chat_tool.utils.gitignore import is_ignored, load_gitignore

_ALWAYS_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}


def discover_python_files(
    repo_root: Path, index_dir_name: str = ".codebase_chat_tool"
) -> list[Path]:
    """Recursively find .py files under repo_root, respecting .gitignore."""
    spec = load_gitignore(repo_root)
    skip_dirs = _ALWAYS_SKIP_DIRS | {index_dir_name}
    results: list[Path] = []
    for path in sorted(repo_root.rglob("*.py")):
        if any(part in skip_dirs for part in path.relative_to(repo_root).parts):
            continue
        if is_ignored(spec, repo_root, path):
            continue
        results.append(path)
    return results


def path_to_module(repo_root: Path, file_path: Path) -> str:
    """Best-effort dotted module name from a file path relative to repo_root."""
    rel = file_path.resolve().relative_to(repo_root.resolve())
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else rel.stem
