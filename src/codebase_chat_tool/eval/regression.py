from dataclasses import dataclass

# (report section, metric key) pairs worth gating CI on. precision_at_k is
# deliberately excluded -- it's expected to be low with a generous top_k and
# is noisy; recall/groundedness/correctness/caller_f1 are the meaningful
# regression signals for "did a change make the system measurably worse".
CHECKED_METRICS: list[tuple[str, str]] = [
    ("retrieval", "recall_at_k"),
    ("groundedness", "grounded_rate"),
    ("correctness", "keyword_coverage"),
    ("impact", "caller_f1"),
]

# LLM outputs are non-deterministic, so allow some slack before failing the build.
DEFAULT_TOLERANCE = 0.10


@dataclass
class RegressionFailure:
    repo: str
    metric: str
    baseline: float
    current: float

    def __str__(self) -> str:
        return f"{self.repo}.{self.metric}: {self.baseline:.2f} -> {self.current:.2f}"


def check_regression(
    current: list[dict], baseline: list[dict], tolerance: float = DEFAULT_TOLERANCE
) -> list[RegressionFailure]:
    """Compares a fresh eval run against a committed baseline. Returns one
    RegressionFailure per (repo, metric) pair that dropped by more than
    `tolerance`. A repo present in `current` but not in `baseline` is skipped
    (nothing to compare against), not treated as a failure."""
    baseline_by_repo = {r["repo"]: r for r in baseline}
    failures: list[RegressionFailure] = []

    for repo_result in current:
        base = baseline_by_repo.get(repo_result["repo"])
        if base is None:
            continue
        for section, key in CHECKED_METRICS:
            current_value = repo_result.get(section, {}).get(key)
            baseline_value = base.get(section, {}).get(key)
            if current_value is None or baseline_value is None:
                continue
            if current_value < baseline_value - tolerance:
                failures.append(
                    RegressionFailure(
                        repo=repo_result["repo"],
                        metric=f"{section}.{key}",
                        baseline=baseline_value,
                        current=current_value,
                    )
                )
    return failures
