"""Single-round repository Q&A benchmark.

Compares three retrieval tools (grep, LSP, RAG) head-to-head on yes/no
questions about a target code repository, measuring accuracy, token usage and
latency. See :mod:`repoqa.benchmark` for the entry point.
"""
import logging

#: Third-party loggers whose per-request chatter is silenced to WARNING.
_NOISY_LOGGERS = (
    "httpx", "httpx2", "openai", "httpcore", "httpcore2",
    "huggingface_hub", "sentence_transformers", "faiss",
    "transformers", "urllib3", "filelock",
)


def configure_logging(*, level: int = logging.INFO) -> None:
    """Configure root logging for a benchmark run.

    Sets a concise timestamped format and quiets noisy HTTP/ML libraries so the
    console shows only the benchmark's own progress.

    :param level: Logging level for the application loggers.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
