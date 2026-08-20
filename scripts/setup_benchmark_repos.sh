#!/usr/bin/env bash
# Convenience wrapper to pre-fetch the pinned benchmark repos before running
# `codebase-chat-tool eval` (which also does this automatically on demand).
set -euo pipefail

python3 - <<'EOF'
from codebase_chat_tool.eval.benchmark.manifest import load_manifest
from codebase_chat_tool.eval.repo_cache import ensure_benchmark_repo

for repo in load_manifest():
    print(f"Fetching {repo.name} @ {repo.commit[:12]}...")
    path = ensure_benchmark_repo(repo)
    print(f"  -> {path}")

print("Benchmark repos ready.")
EOF
