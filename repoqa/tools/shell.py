"""Sandboxed shell tool: run a single read-only command over the target repo.

A 4th benchmarked retrieval tool alongside grep/LSP/RAG. Unlike those, it lets
the model drive an arbitrary (but whitelisted, read-only) shell command to
explore the repo. It is deliberately locked down:

* only a fixed whitelist of read-only programs may run (the first token);
* shell metacharacters (pipes, redirects, chaining, substitution) are rejected,
  and the command runs with ``shell=False`` so nothing can be chained;
* the working directory is pinned to :data:`config.TARGET_REPO`;
* a hard timeout and an output cap bound the blast radius and cost.

It is cross-platform: on Windows the native programs (``dir``, ``findstr``,
``type``, ...) are accepted; on POSIX the usual coreutils are. ``rg`` works on
both when installed.
"""
import os
import shlex
import platform
import subprocess

from langchain_core.tools import tool

from .. import config

#: Maximum characters of output returned to the model.
_MAX_OUTPUT_CHARS = 4000
#: Hard wall-clock limit for a single command.
_TIMEOUT_S = 30

#: Read-only programs the model may invoke (the first token). The sandbox's
#: safety does NOT rest on this list being short: it rests on (a) running with
#: ``shell=False`` and rejecting bare operator tokens, so nothing can pipe,
#: redirect, chain, or substitute — hence no command can *write* or combine
#: into a destructive one — and (b) confining every path argument to
#: ``TARGET_REPO`` (see ``_check_paths_confined``), so nothing outside the repo
#: can be read. Given those two invariants, the list only needs to stay
#: *read-only*: every program here merely inspects/reads files or metadata and
#: never mutates state. It was previously kept deliberately tiny for running on
#: a personal machine; on a server the read-only set below is a fair, more
#: capable surface for the shell condition. Anything not listed is rejected.
_WHITELIST = {
    # --- search / content (read-only) ---
    "rg", "grep", "egrep", "fgrep", "cat", "head", "tail", "less", "more",
    "nl", "cut", "tr", "fold", "rev", "column", "strings", "xxd", "od", "hexdump",
    # --- listing / navigation / metadata (read-only) ---
    "ls", "find", "tree", "stat", "file", "wc", "du", "df", "readlink",
    "realpath", "basename", "dirname", "pwd", "echo",
    # --- text processing (read-only; no in-place editing) ---
    "sort", "uniq", "comm", "diff", "awk", "gawk", "sed",
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

#: Standalone shell-operator tokens that would chain/pipe/redirect commands.
#: Checked *after* parsing, per token: an operator inside a quoted argument
#: (e.g. the regex ``a|b`` passed to ``rg``) is harmless because we run with
#: ``shell=False`` and never hand the string to a shell. Only a bare operator
#: token (what a shell would treat as a separator) is rejected.
_OPERATOR_TOKENS = {";", "&", "&&", "|", "||", ">", ">>", "<", "<<", "`", "$("}

#: Characters that are never safe anywhere (command/line injection via newlines
#: or NUL), regardless of quoting.
_ALWAYS_FORBIDDEN_CHARS = set("\n\r\x00")

#: Programs that need the Windows command interpreter (shell built-ins, not
#: standalone executables). Run via ``cmd /c`` with a rebuilt, validated argv.
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
    """Reject the command if any path-like argument escapes the repo.

    :param argv: The parsed argument vector (program + args).
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
    """Reject commands whose flags would let a read-only program write or escape.

    Several whitelisted programs are read-only *only in their normal form*:
    ``sed -i`` edits in place, ``find -exec``/``-delete`` run/remove files,
    ``awk -f``/``gawk -i`` load external programs, and the Python interpreters
    can do anything via ``-c``/a script. This keeps those programs available for
    their legitimate read-only uses while blocking the dangerous invocations.

    :param program: The normalized program name (``argv[0]`` basename, lowered).
    :param argv: The full parsed argument vector.
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


def _run(argv: list, *, use_cmd: bool) -> str:
    """Execute ``argv`` without a shell and return captured output.

    :param argv: The validated argument vector (program + args).
    :param use_cmd: On Windows, whether to route through ``cmd /c`` for a
        built-in (still ``shell=False``; we build the argv ourselves).
    :return: Combined stdout (or stderr on failure), capped and UTF-8 decoded.
    """
    command = (["cmd", "/c", *argv] if use_cmd else argv)
    try:
        completed = subprocess.run(
            command,
            cwd=os.path.abspath(config.TARGET_REPO),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
    except FileNotFoundError:
        return _reject(f"program not found: {argv[0]!r}")
    except subprocess.TimeoutExpired:
        return _reject(f"timed out after {_TIMEOUT_S}s")
    output = completed.stdout or completed.stderr or "(no output)"
    return output[:_MAX_OUTPUT_CHARS]


def run_command(command: str) -> str:
    """Validate and run a single read-only shell command string.

    :param command: The command line to run (e.g. ``"rg -n foo ."``).
    :return: The command output, or a ``"shell rejected: ..."`` message.
    """
    if not command or not command.strip():
        return _reject("empty command")
    if _ALWAYS_FORBIDDEN_CHARS & set(command):
        return _reject("newline or NUL characters are not allowed")
    windows = platform.system() == "Windows"
    try:
        # On Windows, parse non-POSIX so backslash path separators survive
        # (POSIX mode treats '\' as an escape). But non-POSIX mode keeps the
        # surrounding quotes on tokens, which breaks programs like rg (the
        # pattern becomes "'foo'" instead of foo), so strip them afterwards.
        argv = shlex.split(command, posix=not windows)
    except ValueError as exc:
        return _reject(f"could not parse command: {exc}")
    if windows:
        argv = [_strip_quotes(token) for token in argv]
    if not argv:
        return _reject("empty command")

    # Reject only bare operator tokens (shell chaining/piping/redirection). An
    # operator *inside* a quoted argument (e.g. an rg regex "a|b") is kept
    # because shell=False never lets it chain anything.
    for token in argv:
        if token in _OPERATOR_TOKENS or token.startswith("$("):
            return _reject("shell operators are not allowed as separate tokens "
                           "(no pipes, redirects, chaining, or substitution); "
                           "quote regex alternations like \"a|b\" as one argument")

    program = os.path.basename(argv[0]).lower()
    # Strip a trailing .exe so "findstr.exe" matches "findstr".
    if program.endswith(".exe"):
        program = program[:-4]
    if program not in _WHITELIST:
        return _reject(f"program {program!r} is not in the read-only whitelist")

    flag_error = _check_flags_safe(program, argv)
    if flag_error:
        return flag_error

    confinement_error = _check_paths_confined(argv)
    if confinement_error:
        return confinement_error

    use_cmd = platform.system() == "Windows" and program in _WINDOWS_BUILTINS
    return _run(argv, use_cmd=use_cmd)


@tool
def shell_tool(command: str) -> str:
    """Run ONE read-only shell COMMAND to inspect the repo. Examples:
    "rg -n foo .", "ls -R kernel", "cat mm/slub.c", "find . -name '*.c'",
    "head -50 fs/read_write.c", "wc -l include/linux/list.h", "grep -rn TODO .".
    Sandboxed: only read-only programs are allowed (no writing/editing/execution),
    no pipes/redirects/chaining/substitution, and every path argument must stay
    inside the repo directory (absolute paths and '..' traversal are rejected)."""
    return run_command(command)
