import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QAQuestion:
    id: str
    question: str
    expected_qualnames: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)


@dataclass
class ImpactQuestion:
    id: str
    target: str
    expected_direct_callers: list[str] = field(default_factory=list)
    expected_tests: list[str] = field(default_factory=list)


def load_questions(path: Path) -> tuple[list[QAQuestion], list[ImpactQuestion]]:
    qa: list[QAQuestion] = []
    impact: list[ImpactQuestion] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        kind = record.pop("type")
        if kind == "qa":
            qa.append(QAQuestion(**record))
        elif kind == "impact":
            impact.append(ImpactQuestion(**record))
        else:
            raise ValueError(f"Unknown question type: {kind!r}")
    return qa, impact
