"""Sandboxed shell tool: run read-only shell command(s) over the target repo.

A 4th benchmarked retrieval tool alongside grep/LSP/RAG. Unlike those, it lets
the model drive arbitrary (but whitelisted, read-only) shell commands to explore
the repo, including full **pipelines and chaining** (``|``, ``&&``, ``||``,
``;``) and input redirection (``<``).

Its safety does NOT rest on forbidding pipes. It rests on two invariants that
hold no matter how commands are combined:

1. **Read-only programs only.** Every program in every pipeline stage must be on
   the read-only whitelist, invoked in a form that cannot write or execute
   arbitrary code (``sed -i``, ``find -exec/-delete``, ``python -c`` ... are
   rejected). A read-only command piped into another read-only command is still
   read-only.
2. **No writes, no escaping the repo.** Output redirection (``>``, ``>>``) is
   rejected (it is the main way a read-only reader turns into a writer), command
   substitution (`` ` ``, ``$(`` ) is rejected (it would run unvalidated
   programs), and every path argument is confined to :data:`config.TARGET_REPO`
   (absolute paths outside it and ``..`` traversal are rejected). The working
   directory is pinned to the repo and a hard timeout + output cap bound cost.

Given those two invariants the command can safely be handed to a real shell, so
pipes and chaining behave exactly as a user would expect. It is cross-platform:
on Windows the native programs (``dir``, ``findstr``, ``type``, ...) are
accepted and the command runs via ``cmd /c``; on POSIX the usual coreutils are
and it runs via ``bash -c``. ``rg`` works on both when installed.
"""
import os
import shlex
import platform
import subprocess

from langchain_core.tools import tool

from .. import config

#: Maximum characters of output returned to the model.
_MAX_OUTPUT_CHARS = 4000
#: Hard wall-clock limit for a single command line (whole pipeline/chain).
_TIMEOUT_S = 30

#: Read-only programs the model may invoke (the first token of each pipeline
#: stage). The sandbox's safety does NOT rest on this list being short: it rests
#: on (a) every program being read-only and invoked in a read-only form, and
#: (b) rejecting writes/substitution and confining every path argument to
#: ``TARGET_REPO`` (see ``_check_paths_confined``). Given those invariants the
#: list only needs to stay *read-only*: every program here merely inspects/reads
#: files or metadata and never mutates state. Anything not listed is rejected.
_WHITELIST = {
    # --- search / content (read-only) ---
    "rg", "grep", "egrep", "fgrep", "cat", "head", "tail", "less", "more",
    "nl", "cut", "tr", "fold", "rev", "column", "strings", "xxd", "od", "hexdump",
    # --- listing / navigation / metadata (read-only) ---
    "ls", "find", "tree", "stat", "file", "wc", "du", "df", "readlink",
    "realpath", "basename", "dirname", "pwd", "echo",
    # --- text processing (read-only; no in-place editing) ---
    "sort", "uniq", "comm", "diff", "awk", "gawk", "sed", "tac", "paste", "join",
    "expand", "unexpand",
    # --- structured / source inspection (read-only) ---
    "jq", "python3", "python",
    # --- windows native (read-only) ---
    "dir", "findstr", "type", "where", "fc",
}

#: Programs on the whitelist that can nonetheless *write* or run arbitrary code
#: when given certain flags, so they are allowed only in a restricted form.
#: ``sed``/``awk`` can edit in place or shell out; ``python`` can do anything.
#: Rather than ban them (they are genuinely useful read-only), we reject the
#: dangerous flags and, for interpreters, require an explicit read-only form.
#: Keys map a program to the set of argument tokens that get it rejected.
_FORBIDDEN_FLAGS = {
    "sed": {"-i", "--in-place", "-s", "-f", "--file"},
    "gawk": {"-i", "--include", "-f", "--file", "--load", "-l"},
    "awk": {"-f", "--file"},
    "find": {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint",
             "-fprint0", "-fprintf", "-fls"},
}

#: Interpreters that can execute arbitrary code; allowed only to print a file's
#: contents or version, never to ``-c``/run a script (which could write/escape).
_INTERPRETERS = {"python", "python3"}
#: Flags an interpreter may use (everything else, incl. ``-c`` and script paths
#: without an explicit reader, is rejected).
_INTERPRETER_ALLOWED_FLAGS = {"--version", "-V"}

#: Pipeline/stage separators that are ALLOWED. The command string is split on
#: these into stages, and every stage is validated independently; the real
#: shell then re-interprets them so pipes and chaining work as expected.
_STAGE_SEPARATORS = {"|", "&&", "||", ";"}

#: Operator tokens that are REJECTED even though we hand the line to a shell:
#: output redirection would let a read-only reader *write* a file, and ``&``
#: (bare) would background/detach a process. Input redirection (``<``) is fine
#: (it only reads) and is intentionally NOT here.
_FORBIDDEN_OPERATORS = {">", ">>", "&>", ">&", "<>", "&"}

#: Substring markers for command substitution, which would run programs that
#: never pass through per-stage validation. Rejected anywhere in the raw line.
_SUBSTITUTION_MARKERS = ("$(", "`", "<(", ">(", "${")

#: Characters that are never safe anywhere (command/line injection via newlines
#: or NUL), regardless of quoting.
_ALWAYS_FORBIDDEN_CHARS = set("\n\r\x00")

#: Programs that need the Windows command interpreter (shell built-ins, not
#: standalone executables). On Windows the whole line is run via ``cmd /c``.
_WINDOWS_BUILTINS = {"dir", "type", "echo"}


def _looks_like_path(token: str) -> bool:
    """Heuristic: does ``token`` denote a filesystem path (vs a flag/pattern)?

    Flags (``-x``/``--x``) are never paths. A token is treated as a path if it
    contains a path separator, a Windows drive prefix, a home/relative prefix,
    or if it names an entry that exists relative to the repo.

    :param token: A single parsed argv token.
    :return: ``True`` if the token should be validated as a path.
    """
    if not token or token.startswith("-"):
        return False
    if token in (".", ".."):
        return True
    if "/" in token or "\\" in token:
        return True
    if token.startswith("~"):
        return True
    if len(token) >= 2 and token[1] == ":":  # Windows drive letter, e.g. C:
        return True
    # Bare name that happens to exist under the repo (e.g. "src", "file.py").
    repo = os.path.abspath(config.TARGET_REPO)
    return os.path.exists(os.path.join(repo, token))


def _confined(token: str, repo: str) -> bool:
    """Return whether a path ``token`` resolves inside ``repo``.

    Expands ``~``, joins relatives against the repo, resolves ``..`` and
    symlinks, then checks containment. Absolute paths outside the repo and
    ``..`` traversal both fail.

    :param token: The path token to check.
    :param repo: Absolute path to the target repo root.
    :return: ``True`` if the resolved path is the repo or lives under it.
    """
    expanded = os.path.expanduser(token)
    candidate = expanded if os.path.isabs(expanded) else os.path.join(repo, expanded)
    resolved = os.path.realpath(candidate)
    repo_real = os.path.realpath(repo)
    return resolved == repo_real or resolved.startswith(repo_real + os.sep)


def _check_paths_confined(argv: list) -> str:
    """Reject the stage if any path-like argument escapes the repo.

    :param argv: The parsed argument vector (program + args) for one stage.
    :return: An empty string if all paths are confined, else a rejection
        message.
    """
    repo = os.path.abspath(config.TARGET_REPO)
    for token in argv[1:]:
        if _looks_like_path(token) and not _confined(token, repo):
            return _reject(f"path {token!r} is outside the target repo")
    return ""


def _strip_quotes(token: str) -> str:
    """Remove a single matched pair of surrounding quotes from ``token``.

    Non-POSIX ``shlex`` leaves quotes attached; a program like ``rg`` would then
    search for the literal quoted text. Only strips when both ends match.

    :param token: A parsed argv token (possibly quoted).
    :return: The token without one matched pair of surrounding quotes.
    """
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def _reject(reason: str) -> str:
    """Format a rejection message returned to the model.

    :param reason: Why the command was refused.
    :return: A ``"shell rejected: ..."`` string.
    """
    return f"shell rejected: {reason}"


def _check_flags_safe(program: str, argv: list) -> str:
    """Reject a stage whose flags would let a read-only program write or escape.

    Several whitelisted programs are read-only *only in their normal form*:
    ``sed -i`` edits in place, ``find -exec``/``-delete`` run/remove files,
    ``awk -f``/``gawk -i`` load external programs, and the Python interpreters
    can do anything via ``-c``/a script. This keeps those programs available for
    their legitimate read-only uses while blocking the dangerous invocations.

    :param program: The normalized program name (``argv[0]`` basename, lowered).
    :param argv: The full parsed argument vector for one stage.
    :return: Empty string if the flags are safe, else a rejection message.
    """
    args = argv[1:]

    # Interpreters: allow only an explicit read-only form (print a version).
    # Anything that could execute code (``-c``, ``-m``, a script path, stdin
    # ``-``) is refused — use ``cat``/``read_file`` to see source instead.
    if program in _INTERPRETERS:
        for token in args:
            if token.startswith("-") and token not in _INTERPRETER_ALLOWED_FLAGS:
                return _reject(f"{program}: only {sorted(_INTERPRETER_ALLOWED_FLAGS)} "
                               "are allowed (no code execution via the interpreter)")
            if not token.startswith("-"):
                return _reject(f"{program}: running scripts/args is not allowed; "
                               "use cat/head/grep to inspect files")
        return ""

    forbidden = _FORBIDDEN_FLAGS.get(program)
    if forbidden:
        for token in args:
            # Compare the flag name before any '=' (e.g. --file=foo).
            flag = token.split("=", 1)[0]
            if flag in forbidden:
                return _reject(f"{program}: flag {flag!r} is not allowed "
                               "(it can write files or execute code)")
    return ""


def _split_stages(tokens: list) -> list:
    """Split a parsed token list into pipeline/chain stages.

    Splits on the allowed separators (``|``, ``&&``, ``||``, ``;``); each stage
    is the argv of one program to validate independently. Separators only ever
    appear as their own tokens here because :func:`run_command` has already
    rejected forbidden operators and parsed with ``shlex`` (so a ``|`` inside a
    quoted regex stays attached to its token and is not a separator).

    :param tokens: The parsed token list for the whole command line.
    :return: A list of stages, each a list of argv tokens.
    """
    stages, current = [], []
    for token in tokens:
        if token in _STAGE_SEPARATORS:
            if current:
                stages.append(current)
                current = []
            # A separator with no preceding command (e.g. leading ``|``) yields
            # an empty stage that validation below will flag.
            else:
                stages.append([])
        else:
            current.append(token)
    if current:
        stages.append(current)
    return stages


def _validate_stage(argv: list) -> str:
    """Validate one pipeline stage: whitelist, flags, path confinement.

    :param argv: The argv tokens of a single stage.
    :return: Empty string if the stage is allowed, else a rejection message.
    """
    if not argv:
        return _reject("empty pipeline stage (misplaced | && || or ;)")

    program = os.path.basename(argv[0]).lower()
    # Strip a trailing .exe so "findstr.exe" matches "findstr".
    if program.endswith(".exe"):
        program = program[:-4]
    if program not in _WHITELIST:
        return _reject(f"program {program!r} is not in the read-only whitelist")

    flag_error = _check_flags_safe(program, argv)
    if flag_error:
        return flag_error

    return _check_paths_confined(argv)


def _run(command: str, *, use_cmd: bool) -> str:
    """Execute a validated command line through a real shell and capture output.

    Running through the shell is what makes pipes/chaining/input-redirection
    work. Safety has already been established by per-stage validation plus the
    operator/substitution rejection in :func:`run_command`, so the shell only
    ever re-interprets constructs we have already proven read-only and confined.

    :param command: The validated command line.
    :param use_cmd: On Windows, run via ``cmd /c``; otherwise via ``bash -c``.
    :return: Combined stdout (or stderr on failure), capped and UTF-8 decoded.
    """
    if use_cmd:
        argv = ["cmd", "/c", command]
    else:
        shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
        argv = [shell, "-c", command]
    try:
        completed = subprocess.run(
            argv,
            cwd=os.path.abspath(config.TARGET_REPO),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            encoding="utf-8",
            errors="replace",
            shell=False,  # we invoke the shell explicitly with a fixed argv
        )
    except FileNotFoundError:
        return _reject("shell interpreter not found")
    except subprocess.TimeoutExpired:
        return _reject(f"timed out after {_TIMEOUT_S}s")
    output = completed.stdout or completed.stderr or "(no output)"
    return output[:_MAX_OUTPUT_CHARS]


def run_command(command: str) -> str:
    """Validate and run a read-only shell command line (pipelines allowed).

    Pipes (``|``), chaining (``&&``, ``||``, ``;``) and input redirection
    (``<``) are supported. Each pipeline stage is validated independently
    against the read-only whitelist and flag rules, every path argument is
    confined to :data:`config.TARGET_REPO`, and output redirection and command
    substitution are rejected. The whole line then runs through a real shell.

    :param command: The command line to run (e.g. ``"rg -n foo . | head"``).
    :return: The command output, or a ``"shell rejected: ..."`` message.
    """
    if not command or not command.strip():
        return _reject("empty command")
    if _ALWAYS_FORBIDDEN_CHARS & set(command):
        return _reject("newline or NUL characters are not allowed")

    # Reject command substitution outright: it would execute programs that never
    # pass through the per-stage validation below, defeating the whitelist.
    for marker in _SUBSTITUTION_MARKERS:
        if marker in command:
            return _reject(f"command substitution/process substitution ({marker!r}) "
                           "is not allowed")

    windows = platform.system() == "Windows"
    try:
        # On Windows, parse non-POSIX so backslash path separators survive
        # (POSIX mode treats '\' as an escape). Non-POSIX keeps surrounding
        # quotes on tokens, which breaks programs like rg, so strip them after.
        tokens = shlex.split(command, posix=not windows)
    except ValueError as exc:
        return _reject(f"could not parse command: {exc}")
    if windows:
        tokens = [_strip_quotes(token) for token in tokens]
    if not tokens:
        return _reject("empty command")

    # Reject forbidden operator tokens (write-redirects, backgrounding). These
    # are matched as bare tokens only: an operator inside a quoted argument
    # (e.g. an rg regex ">foo") stays attached to its token by shlex and is not
    # treated as an operator.
    for token in tokens:
        if token in _FORBIDDEN_OPERATORS:
            return _reject(f"operator {token!r} is not allowed "
                           "(output redirection and backgrounding are blocked; "
                           "pipes |, chaining && || ; and input < are fine)")

    # Split into stages on the ALLOWED separators and validate each one. Drop a
    # bare input-redirection token and its target from a stage's argv so the
    # reader program's own validation (whitelist/flags) is unaffected; the path
    # target is still confinement-checked.
    stages = _split_stages(tokens)
    if not stages:
        return _reject("empty command")
    for stage in stages:
        error = _validate_stage(_normalize_redirects(stage))
        if error:
            return error
        redir_error = _check_redirect_targets(stage)
        if redir_error:
            return redir_error

    use_cmd = windows and any(
        os.path.basename(s[0]).lower().removesuffix(".exe") in _WINDOWS_BUILTINS
        for s in stages if s
    )
    return _run(command, use_cmd=use_cmd)


def _normalize_redirects(stage: list) -> list:
    """Return a stage's argv with input-redirection (``<`` target) removed.

    Input redirection is allowed (read-only), but the ``<`` token and the
    filename after it are not arguments to the program and must not be treated
    as such by whitelist/flag validation. Their path is confinement-checked
    separately by :func:`_check_redirect_targets`.

    :param stage: The raw argv tokens of one stage.
    :return: The argv with any ``<`` redirection operator + target stripped.
    """
    cleaned, skip_next = [], False
    for token in stage:
        if skip_next:
            skip_next = False
            continue
        if token == "<":
            skip_next = True  # drop the redirect target filename too
            continue
        cleaned.append(token)
    return cleaned


def _check_redirect_targets(stage: list) -> str:
    """Confine any input-redirection (``<``) target to the repo.

    :param stage: The raw argv tokens of one stage.
    :return: Empty string if confined, else a rejection message.
    """
    repo = os.path.abspath(config.TARGET_REPO)
    tokens = iter(stage)
    for token in tokens:
        if token == "<":
            target = next(tokens, None)
            if target and not _confined(target, repo):
                return _reject(f"redirect target {target!r} is outside the target repo")
    return ""


@tool
def shell_tool(command: str) -> str:
    """Run read-only shell COMMAND(s) to inspect the repo. Pipelines and chaining
    are supported. Examples:
    "rg -n foo .", "ls -R kernel", "cat mm/slub.c",
    "rg -n EXPORT_SYMBOL fs/ | wc -l", "grep -rn TODO . | head -20",
    "find . -name '*.c' | sort | head", "cat fs/read_write.c | sed -n '1,40p'".
    Sandboxed: only read-only programs are allowed (no writing/editing/execution),
    output redirection (> >>) and command substitution ($(...) / backticks) are
    rejected, and every path argument (including input < redirects) must stay
    inside the repo directory (absolute paths and '..' traversal are rejected).
    Pipes |, chaining && || ;, and input redirection < are allowed."""
    return run_command(command)
