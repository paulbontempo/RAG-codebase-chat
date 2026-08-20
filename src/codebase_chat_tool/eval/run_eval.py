import json
from pathlib import Path

from rich.console import Console

from codebase_chat_tool.agent.chat_graph import run_chat_turn
from codebase_chat_tool.agent.context import load_repo_context
from codebase_chat_tool.config import Settings, get_settings
from codebase_chat_tool.eval.benchmark.manifest import BenchmarkRepo, load_manifest
from codebase_chat_tool.eval.correctness import aggregate_correctness_scores, score_keyword_coverage
from codebase_chat_tool.eval.groundedness import aggregate_groundedness_scores, score_groundedness
from codebase_chat_tool.eval.impact_eval import aggregate_impact_scores, evaluate_impact_question
from codebase_chat_tool.eval.questions import load_questions
from codebase_chat_tool.eval.regression import check_regression
from codebase_chat_tool.eval.repo_cache import ensure_benchmark_repo
from codebase_chat_tool.eval.retrieval_metrics import aggregate_retrieval_scores, score_retrieval
from codebase_chat_tool.ingestion.pipeline import run_indexing
from codebase_chat_tool.llm.factory import get_provider

console = Console()

BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark"


def _evaluate_repo(repo: BenchmarkRepo, settings: Settings) -> dict:
    console.print(f"[cyan]Preparing benchmark repo: {repo.name}[/cyan]")
    package_path = ensure_benchmark_repo(repo)

    console.print(f"[cyan]Indexing {repo.name}...[/cyan]")
    run_indexing(str(package_path))

    ctx = load_repo_context(package_path, settings)
    provider = get_provider(settings)

    questions_path = BENCHMARK_DIR / f"questions_{repo.name}.jsonl"
    qa_questions, impact_questions = load_questions(questions_path)

    retrieval_scores, groundedness_scores, correctness_scores = [], [], []
    console.print(f"[cyan]Running {len(qa_questions)} QA questions against {repo.name}...[/cyan]")
    for q in qa_questions:
        retrieved = ctx.retriever.search(q.question, top_k=settings.retrieval_top_k)
        retrieval_scores.append(
            score_retrieval([r.chunk.qualname for r in retrieved], q.expected_qualnames)
        )

        history = run_chat_turn(ctx, provider, [], q.question)
        answer = history[-1].content
        groundedness_scores.append(score_groundedness(answer, history))
        correctness_scores.append(score_keyword_coverage(answer, q.expected_keywords))

    console.print(
        f"[cyan]Running {len(impact_questions)} blast-radius questions "
        f"against {repo.name}...[/cyan]"
    )
    impact_results = [
        evaluate_impact_question(ctx, q.target, q.expected_direct_callers, q.expected_tests)
        for q in impact_questions
    ]

    return {
        "repo": repo.name,
        "commit": repo.commit,
        "n_qa_questions": len(qa_questions),
        "n_impact_questions": len(impact_questions),
        "retrieval": aggregate_retrieval_scores(retrieval_scores),
        "groundedness": aggregate_groundedness_scores(groundedness_scores),
        "correctness": aggregate_correctness_scores(correctness_scores),
        "impact": aggregate_impact_scores(impact_results),
    }


def _render_markdown(repo_results: list[dict]) -> str:
    lines = ["# Evaluation report", ""]
    lines.append(
        "All metrics below are computed deterministically (exact set/string comparisons) "
        "-- no LLM-as-judge step is used anywhere in this report. "
        "See `docs/architecture.md` for what each metric measures."
    )
    lines.append("")
    lines.append(
        "| Repo | P@k | R@k | MRR | Citation rate | Grounded rate | Keyword coverage "
        "| Caller P | Caller R | Caller F1 | Test F1 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in repo_results:
        ret, gr, co, im = r["retrieval"], r["groundedness"], r["correctness"], r["impact"]
        lines.append(
            f"| {r['repo']} "
            f"| {ret['precision_at_k']:.2f} | {ret['recall_at_k']:.2f} | {ret['mrr']:.2f} "
            f"| {gr['citation_rate']:.2f} | {gr['grounded_rate']:.2f} "
            f"| {co['keyword_coverage']:.2f} "
            f"| {im['caller_precision']:.2f} | {im['caller_recall']:.2f} | {im['caller_f1']:.2f} "
            f"| {im.get('test_f1', 0.0):.2f} |"
        )
    lines.append("")
    for r in repo_results:
        lines.append(f"## {r['repo']} (@ `{r['commit'][:12]}`)")
        lines.append(
            f"- {r['n_qa_questions']} QA questions, "
            f"{r['n_impact_questions']} blast-radius questions"
        )
        lines.append(f"- Retrieval: {r['retrieval']}")
        lines.append(f"- Groundedness: {r['groundedness']}")
        lines.append(f"- Correctness: {r['correctness']}")
        lines.append(f"- Impact: {r['impact']}")
        lines.append("")
    return "\n".join(lines)


DEFAULT_BASELINE_PATH = Path("docs/eval_baseline.json")


def run_eval(
    repo_names: list[str] | None = None,
    report_path: Path = Path("docs/eval_report.md"),
    results_path: Path = Path("eval_results.json"),
    baseline_path: Path | None = DEFAULT_BASELINE_PATH,
    update_baseline: bool = False,
) -> list[dict]:
    settings = get_settings()
    manifest = load_manifest()
    repos = [r for r in manifest if repo_names is None or r.name in repo_names]
    if not repos:
        console.print(f"[red]No matching benchmark repos in {[r.name for r in manifest]}[/red]")
        raise SystemExit(1)

    repo_results = [_evaluate_repo(repo, settings) for repo in repos]

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(repo_results, indent=2))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_markdown(repo_results))

    console.print(f"[green]Wrote {results_path} and {report_path}[/green]")

    if baseline_path is not None and update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(repo_results, indent=2))
        console.print(f"[green]Updated baseline at {baseline_path}[/green]")
    elif baseline_path is not None and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        failures = check_regression(repo_results, baseline)
        if failures:
            console.print(f"[red]Eval regression detected ({len(failures)} metric(s)):[/red]")
            for f in failures:
                console.print(f"  [red]{f}[/red]")
            raise SystemExit(1)
        console.print(f"[green]No regression vs. baseline ({baseline_path})[/green]")

    return repo_results
