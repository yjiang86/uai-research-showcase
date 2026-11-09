"""An expression budget is not an evidence-storage capacity."""
from __future__ import annotations

from .config import RuntimeConfig
from .signals import priority
from .stance import currently_valid
from .types import Tier, Trace


def expressed(trace: Trace) -> bool:
    return (currently_valid(trace) and trace.tier != Tier.COLD and trace.merged_into is None
            and not trace.suppressed)


def usage(traces: dict[str, Trace], config: RuntimeConfig) -> dict[str, int]:
    lines = [f"[{key}] {t.current.text}" for key, t in traces.items() if expressed(t)]
    text = "\n".join(lines)
    return {"records": sum(bool(line.strip()) for line in text.splitlines()),
            "words": len(text.split()), "bytes": len(text.encode("utf-8")),
            "record_budget": config.reader_records, "word_budget": config.reader_words,
            "byte_budget": config.reader_bytes}


def fits(value: dict[str, int]) -> bool:
    return all(value[n] <= value[b] for n, b in (("records", "record_budget"),
               ("words", "word_budget"), ("bytes", "byte_budget")))


def victims(traces: dict[str, Trace], operation: int, config: RuntimeConfig) -> list[str]:
    indegree = {k: 0 for k in traces}
    for t in traces.values():
        if currently_valid(t):
            for dep in t.depends_on:
                if dep in indegree:
                    indegree[dep] += 1
    return sorted((k for k, t in traces.items() if expressed(t)),
                  key=lambda k: (priority(traces[k], operation, indegree[k], config), k))
