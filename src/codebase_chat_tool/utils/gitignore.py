from pathlib import Path

import pathspec


def load_gitignore(repo_root: Path) -> pathspec.PathSpec:
    patterns: list[str] = [".git/"]
    gitignore_path = repo_root / ".gitignore"
    if gitignore_path.exists():
        patterns.extend(gitignore_path.read_text().splitlines())
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def is_ignored(spec: pathspec.PathSpec, repo_root: Path, path: Path) -> bool:
    return spec.match_file(str(path.relative_to(repo_root)))
