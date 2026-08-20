import json
import re
from dataclasses import dataclass

from codebase_chat_tool.llm.base import Message

_CITATION_RE = re.compile(r"([\w./\\-]+\.py):(\d+)")


@dataclass
class GroundednessScore:
    citation_count: int
    grounded_count: int

    @property
    def rate(self) -> float:
        """Fraction of citations that point at a location actually surfaced by a
        tool call during this conversation. 1.0 if there were no citations at all
        is intentionally NOT returned here -- see `has_citations`."""
        return self.grounded_count / self.citation_count if self.citation_count else 0.0

    @property
    def has_citations(self) -> bool:
        return self.citation_count > 0


def _seen_ranges(messages: list[Message]) -> list[tuple[str, int, int]]:
    """Extract (file, start_line, end_line) ranges from every tool-result message
    in the conversation -- i.e. every location the model was actually shown."""
    ranges: list[tuple[str, int, int]] = []
    for msg in messages:
        if msg.role != "tool":
            continue
        try:
            payload = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if isinstance(entry, dict) and "file" in entry and "start_line" in entry:
                ranges.append(
                    (entry["file"], entry["start_line"], entry.get("end_line", entry["start_line"]))
                )
    return ranges


def score_groundedness(answer_text: str, messages: list[Message]) -> GroundednessScore:
    """Mechanically checks each `file.py:line` citation in the answer against the
    tool-call results actually returned during the conversation -- i.e. did the
    model cite something it was really shown, rather than trusting an LLM's
    self-report of faithfulness."""
    citations = _CITATION_RE.findall(answer_text)
    if not citations:
        return GroundednessScore(citation_count=0, grounded_count=0)

    seen = _seen_ranges(messages)
    grounded = 0
    for file_part, line_str in citations:
        line = int(line_str)
        for seen_file, start, end in seen:
            if seen_file.endswith(file_part) or file_part.endswith(seen_file):
                if start <= line <= end:
                    grounded += 1
                    break

    return GroundednessScore(citation_count=len(citations), grounded_count=grounded)


def aggregate_groundedness_scores(scores: list[GroundednessScore]) -> dict[str, float]:
    if not scores:
        return {"citation_rate": 0.0, "grounded_rate": 0.0}
    with_citations = [s for s in scores if s.has_citations]
    return {
        "citation_rate": len(with_citations) / len(scores),
        "grounded_rate": (sum(s.rate for s in with_citations) / len(with_citations))
        if with_citations
        else 0.0,
    }
