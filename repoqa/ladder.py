"""Subsystem-ladder scoping for the scaling experiment.

The ladder (``ladder-fixed.yaml``) defines *rungs*: each rung asks the SAME
fixed question set while the INDEXED codebase grows (rung 1 = a few subsystems,
rung 3 = most of the kernel). Comparing a tool across rungs isolates codebase
size as the only variable — "how does accuracy/cost change as the haystack
grows?".

A rung's indexed codebase is the union of its subsystems' path patterns, taken
from ``subsystems.yaml``. This module turns that into a concrete, canonical set
of ``.c``/``.h`` files and installs it as the process-wide **active scope** that
the search/read tools consult, so a tool can never see a file outside the rung.

Containment model (decision ii): a path that exists in the checkout but is NOT
in the active scope is treated by the tools exactly as if it did not exist, so a
reduced rung is indistinguishable from a kernel that only ever contained those
subsystems — no signal leaks that a larger tree is present.

Path-pattern semantics (from ``subsystems.yaml`` notes):

* a pattern ending in ``/`` covers that whole subtree;
* ``*`` and ``?`` are shell wildcards that never cross ``/``;
* otherwise the pattern names a single file.

All matched paths are intersected with :data:`config.SOURCE_EXTENSIONS` (the
``.c``/``.h`` question scope) and resolved to realpaths for robust membership
tests.
"""
import os
import glob
import fnmatch
import logging

from . import config

log = logging.getLogger("repoqa.ladder")

#: Process-wide active scope: a set of canonical (realpath) files the tools may
#: see, or ``None`` meaning "no ladder scope active" (tools fall back to walking
#: the whole ``TARGET_REPO`` as before). Set via :func:`activate_scope`.
_ACTIVE_SCOPE = None
#: Human-readable label of the active rung (for logging / records).
_ACTIVE_LABEL = None


def _load_yaml(path: str):
    """Load a YAML file.

    :param path: Path to the YAML document.
    :return: The parsed Python object.
    """
    import yaml
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_subsystems(subsystems_path: str) -> dict:
    """Load ``subsystems.yaml`` into a name -> subsystem-dict mapping.

    :param subsystems_path: Path to ``subsystems.yaml``.
    :return: Mapping of subsystem name to its record (which holds ``paths``).
        Insertion order follows the YAML (priority) order.
    """
    data = _load_yaml(subsystems_path)
    return {s["name"]: s for s in data.get("subsystems", [])}


def load_subsystems_ordered(subsystems_path: str) -> list:
    """Load ``subsystems.yaml`` as the ORDERED list of subsystem records.

    The order is the priority order that defines the file partition (see
    :func:`build_partition`), so callers that resolve a rung's scope must use
    this, not an arbitrary mapping iteration.

    :param subsystems_path: Path to ``subsystems.yaml``.
    :return: List of subsystem dicts in YAML order.
    """
    data = _load_yaml(subsystems_path)
    return list(data.get("subsystems", []))


def load_ladder(ladder_path: str) -> list:
    """Load a ladder YAML into a list of rung dicts (in order).

    :param ladder_path: Path to ``ladder-fixed.yaml`` (or similar).
    :return: List of rung dicts, each with ``name``/``subsystems``/``questions``.
    """
    return _load_yaml(ladder_path)


def get_rung(ladder_path: str, rung_name: str) -> dict:
    """Return a single rung by name from a ladder file.

    :param ladder_path: Path to the ladder YAML.
    :param rung_name: The rung's ``name`` (e.g. ``"ladder_1"``).
    :return: The rung dict.
    :raises ValueError: If no rung with that name exists.
    """
    for rung in load_ladder(ladder_path):
        if rung.get("name") == rung_name:
            return rung
    names = [r.get("name") for r in load_ladder(ladder_path)]
    raise ValueError(f"rung {rung_name!r} not found; available: {names}")


def _matches_source(path: str) -> bool:
    """Return whether ``path`` is an in-scope source file (``.c``/``.h``).

    :param path: A filesystem path.
    :return: ``True`` if it ends with a configured source extension.
    """
    return path.endswith(tuple(config.SOURCE_EXTENSIONS))


def _expand_pattern(repo: str, pattern: str):
    """Expand one subsystem path pattern into concrete source files.

    Honors the ``subsystems.yaml`` semantics: a trailing ``/`` means the whole
    subtree; ``*``/``?`` are wildcards that do not cross ``/``; otherwise the
    pattern is a single file. Only existing ``.c``/``.h`` files are yielded;
    patterns that match nothing in this checkout are silently skipped.

    :param repo: Absolute path to the kernel checkout root.
    :param pattern: A path pattern relative to ``repo``.
    :yield: Absolute paths of matched source files.
    """
    pattern = pattern.strip()
    if not pattern:
        return
    has_wildcard = any(ch in pattern for ch in "*?[")
    if pattern.endswith("/"):
        base_pattern = pattern.rstrip("/")
        if has_wildcard:
            # e.g. ``fs/*/`` — glob the directory level(s), then walk each match
            # as a subtree. ``*`` must not cross ``/`` (glob honors this).
            for base in glob.glob(os.path.join(repo, base_pattern)):
                if os.path.isdir(base):
                    for root, _dirs, files in os.walk(base):
                        for name in files:
                            if _matches_source(name):
                                yield os.path.join(root, name)
            return
        # Plain subtree: walk it and yield every source file beneath.
        base = os.path.join(repo, base_pattern)
        if os.path.isdir(base):
            for root, _dirs, files in os.walk(base):
                for name in files:
                    if _matches_source(name):
                        yield os.path.join(root, name)
        return
    full = os.path.join(repo, pattern)
    if has_wildcard:
        # Wildcard that must not cross '/': glob (non-recursive across dirs).
        for match in glob.glob(full):
            if os.path.isfile(match) and _matches_source(match):
                yield match
            elif os.path.isdir(match):
                # A wildcard directory match: include its direct source files.
                for root, _dirs, files in os.walk(match):
                    for name in files:
                        if _matches_source(name):
                            yield os.path.join(root, name)
        return
    # Exact file.
    if os.path.isfile(full) and _matches_source(full):
        yield full


def build_partition(repo: str, subsystems_ordered: list) -> dict:
    """Assign every source file to exactly ONE subsystem (priority partition).

    ``subsystems.yaml`` defines a partition: subsystems are listed in priority
    order and a file belongs to the FIRST subsystem whose path pattern matches
    it. A broad pattern (e.g. ``fs/`` under ``vfs``) therefore only receives
    files no earlier subsystem (``fs-exec``, ``fs-ext4`` ...) already claimed.
    Resolving each subsystem independently would double-count those files and
    massively inflate a rung's scope, so the assignment MUST be global and
    order-sensitive.

    :param repo: Absolute path to the kernel checkout root.
    :param subsystems_ordered: The subsystem dicts in their YAML order.
    :return: Mapping of canonical (realpath) source file -> owning subsystem
        name (first match wins).
    """
    repo = os.path.abspath(repo)
    owner = {}
    for sub in subsystems_ordered:
        name = sub["name"]
        for pattern in sub.get("paths", []):
            for path in _expand_pattern(repo, pattern):
                real = os.path.realpath(path)
                # First match wins: do not overwrite an earlier claim.
                owner.setdefault(real, name)
    return owner


def resolve_rung_files(repo: str, rung: dict, subsystems: dict,
                       subsystems_ordered: list = None) -> set:
    """Resolve a rung's subsystems to a canonical set of source-file realpaths.

    Uses the global priority partition (:func:`build_partition`) so each file is
    attributed to exactly one subsystem; the rung's scope is every file whose
    owning subsystem is in the rung. This matches the ``subsystems.yaml``
    definition and the declared per-rung ``ch_files``/``ch_loc`` sizes.

    :param repo: Absolute path to the kernel checkout root.
    :param rung: A rung dict (with a ``subsystems`` name list).
    :param subsystems: Mapping from :func:`load_subsystems` (name -> record).
    :param subsystems_ordered: The subsystem dicts in YAML order (required for a
        correct partition). If omitted, falls back to ``subsystems.values()``
        order, which is insertion-ordered in Python 3.7+ and matches the file.
    :return: Set of canonical ``.c``/``.h`` files in the rung's scope.
    """
    if subsystems_ordered is None:
        subsystems_ordered = list(subsystems.values())
    rung_names = set(rung.get("subsystems", []))
    missing = [n for n in rung_names if n not in subsystems]
    if missing:
        log.warning("ladder: %d subsystem(s) in rung %r not found in "
                    "subsystems.yaml: %s", len(missing), rung.get("name"), missing)
    owner = build_partition(repo, subsystems_ordered)
    return {path for path, sub in owner.items() if sub in rung_names}


def filter_questions(all_questions: list, rung: dict) -> list:
    """Return the rung's questions, in the rung's id order.

    :param all_questions: All question dicts loaded from the ``.jsonl``.
    :param rung: A rung dict (with a ``questions`` id list).
    :return: The subset of ``all_questions`` whose ``id`` is in the rung, ordered
        as the rung lists them. Ids absent from the jsonl are skipped (warned).
    """
    by_id = {q.get("id"): q for q in all_questions}
    selected = []
    missing = []
    for qid in rung.get("questions", []):
        q = by_id.get(qid)
        if q is None:
            missing.append(qid)
        else:
            selected.append(q)
    if missing:
        log.warning("ladder: %d question id(s) in rung %r not found in the "
                    "question set: %s", len(missing), rung.get("name"), missing)
    return selected


def activate_scope(files: set, *, label: str = None) -> None:
    """Install ``files`` as the process-wide active tool scope.

    After this, the search/read tools restrict themselves to exactly these
    files; anything else is invisible (treated as nonexistent).

    :param files: Set of canonical (realpath) files the tools may see.
    :param label: Optional human-readable rung label (for logs/records).
    """
    global _ACTIVE_SCOPE, _ACTIVE_LABEL
    _ACTIVE_SCOPE = set(files)
    _ACTIVE_LABEL = label
    log.info("ladder: activated scope %r with %d files", label, len(_ACTIVE_SCOPE))


def clear_scope() -> None:
    """Remove any active scope (tools revert to the whole ``TARGET_REPO``)."""
    global _ACTIVE_SCOPE, _ACTIVE_LABEL
    _ACTIVE_SCOPE = None
    _ACTIVE_LABEL = None


def active_scope():
    """Return the active scope set, or ``None`` if no ladder scope is active.

    :return: The set of canonical files, or ``None``.
    """
    return _ACTIVE_SCOPE


def active_label():
    """Return the active rung label, or ``None``.

    :return: The label string, or ``None``.
    """
    return _ACTIVE_LABEL


def in_scope(path: str) -> bool:
    """Return whether ``path`` is visible under the active scope.

    When no scope is active, everything is visible (returns ``True``). When a
    scope is active, only its canonical files are visible — this is the single
    containment check the tools share.

    :param path: A filesystem path (absolute or relative to the repo).
    :return: ``True`` if the tool may see the file.
    """
    if _ACTIVE_SCOPE is None:
        return True
    repo = os.path.abspath(config.TARGET_REPO)
    candidate = path if os.path.isabs(path) else os.path.join(repo, path)
    return os.path.realpath(candidate) in _ACTIVE_SCOPE


def scope_file_list() -> list:
    """Return the active scope as a sorted list of file paths (for grep/rg).

    :return: Sorted list of canonical file paths, or ``[]`` if no scope active.
    """
    if _ACTIVE_SCOPE is None:
        return []
    return sorted(_ACTIVE_SCOPE)
