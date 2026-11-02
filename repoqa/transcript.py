"""JSONL transcript logging for a benchmark run.

Each run writes one ``.jsonl`` file: one JSON object per line. Per-question
records capture everything needed to reconstruct the run — the exact prompts,
every agent round's reasoning / tool args / full tool output, the answerer's
context and output, token usage and latency. A final ``summary`` record holds
the aggregated per-tool metrics the plotter consumes.

JSONL (rather than a flat text log) keeps records queryable and avoids the
duplication that made the old text logs balloon to thousands of lines.
"""
import os
import json
from datetime import datetime


class Transcript:
    """Append-only JSONL writer for a single run.

    :param directory: Directory to create the file in (created if missing).
        Ignored when ``path`` is given.
    :param path: Explicit output file path (overrides the timestamped default).
        Its parent directory is created if missing.
    """

    def __init__(self, *, directory: str = "logs", path: str = None):
        if path:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            self.path = path
        else:
            os.makedirs(directory, exist_ok=True)
            self.path = os.path.join(directory, f"run_{datetime.now():%Y%m%d_%H%M%S}.jsonl")

    def record(self, obj: dict) -> None:
        """Append one JSON object as a line.

        :param obj: A JSON-serializable record.
        """
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def meta(self, **fields) -> None:
        """Write a run-level metadata record (``type="meta"``).

        :param fields: Arbitrary metadata (cache mode, round budget, ...).
        """
        self.record({"type": "meta", **fields})

    def question_record(self, *, record: dict) -> None:
        """Write a per-question record (``type="question"``).

        :param record: The fully-populated question record (see
            :func:`repoqa.agent.answer_question`).
        """
        self.record({"type": "question", **record})

    def summary(self, *, cache_mode: str, max_rounds: int, tools: dict) -> None:
        """Write the aggregated per-tool summary record (``type="summary"``).

        :param cache_mode: The active cache mode.
        :param max_rounds: The round budget ``n`` for the run.
        :param tools: Mapping of tool name to its summary dict.
        """
        self.record({"type": "summary", "cache_mode": cache_mode,
                     "max_rounds": max_rounds, "tools": tools})
