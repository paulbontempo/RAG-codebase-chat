# Evaluation report

All metrics below are computed deterministically (exact set/string comparisons) -- no LLM-as-judge step is used anywhere in this report. See `docs/architecture.md` for what each metric measures.

| Repo | P@k | R@k | MRR | Citation rate | Grounded rate | Keyword coverage | Caller P | Caller R | Caller F1 | Test F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| requests | 0.29 | 1.00 | 0.66 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| flask | 0.12 | 0.92 | 0.58 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| typer | 0.10 | 0.67 | 0.33 | 0.50 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 | 0.40 |

## requests (@ `6e83187b8feb`)
- 6 QA questions, 6 blast-radius questions
- Retrieval: {'precision_at_k': 0.2916666666666667, 'recall_at_k': 1.0, 'mrr': 0.6626984126984127}
- Groundedness: {'citation_rate': 0.6666666666666666, 'grounded_rate': 1.0}
- Correctness: {'keyword_coverage': 1.0}
- Impact: {'caller_precision': 1.0, 'caller_recall': 1.0, 'caller_f1': 1.0, 'test_f1': 1.0, 'resolution_rate': 1.0}

## flask (@ `22d924701a6a`)
- 6 QA questions, 6 blast-radius questions
- Retrieval: {'precision_at_k': 0.125, 'recall_at_k': 0.9166666666666666, 'mrr': 0.5833333333333334}
- Groundedness: {'citation_rate': 1.0, 'grounded_rate': 1.0}
- Correctness: {'keyword_coverage': 1.0}
- Impact: {'caller_precision': 1.0, 'caller_recall': 1.0, 'caller_f1': 1.0, 'test_f1': 1.0, 'resolution_rate': 1.0}

## typer (@ `fe2aa0e2f9c8`)
- 6 QA questions, 5 blast-radius questions
- Retrieval: {'precision_at_k': 0.10416666666666667, 'recall_at_k': 0.6666666666666666, 'mrr': 0.32936507936507936}
- Groundedness: {'citation_rate': 0.5, 'grounded_rate': 1.0}
- Correctness: {'keyword_coverage': 0.75}
- Impact: {'caller_precision': 1.0, 'caller_recall': 1.0, 'caller_f1': 1.0, 'test_f1': 0.4, 'resolution_rate': 1.0}
