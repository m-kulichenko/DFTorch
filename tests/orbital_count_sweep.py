"""Re-runnable sweep for hardcoded orbital-count and basis-layout sites.

This module is the machine half of decision **D-04** (see
``.planning/phases/05-regression-safety-and-support-policy-cleanup/05-CONTEXT.md``).
It enumerates every place in ``src/dftorch`` where the code makes a decision from
a hardcoded orbital count, a hardcoded shell count, or a hardcoded per-shell
basis-layout table.  ``docs/ORBITAL-COUNT-INVENTORY.md`` gives each returned
record a written disposition, and ``tests/test_orbital_count_guards.py`` matches
the two in **both** directions, so a site added by a later phase fails the suite
rather than quietly escaping the audit.

Why this is a sweep and not a document
--------------------------------------
An orbital-count site that silently falls through does not raise, does not
produce a NaN, and does not change the shape of anything.  It returns a
well-formed matrix with zeros where the f values belong.  The worked example is
the defect fixed in commit ``4dbffaa``: ``read_skf_table`` inferred shell
presence from ``E_l != 0.0``, mio-1-1's ``H-H.skf`` carries a rounding-noise
placeholder ``Ep = 0.000039`` Hartree, and hydrogen was therefore assigned four
orbitals instead of one.  The phantom p functions picked up real Slater-Koster
overlap, which made the overlap matrix indefinite, ``S^(-1/2)`` undefined and
``2*Tr(D S)`` unequal to the electron count -- and CH4 could never converge.
Every assertion in the suite about finiteness, shape and symmetry passed
throughout.  See ``tests/test_shell_count_parsing.py``.

Why the sweep flattens newlines before matching
-----------------------------------------------
Every match is performed on a copy of the file in which each newline-plus-indent
run has been collapsed to a single space, so a comparison that a formatter has
wrapped is still a single contiguous match.

Measured on live code (2026-08-03), applying the pair-level form
``(a == x) & (b == y)`` -- the shape the plan-time research swept for -- to the
real ``_h0ands.py``: **9** hits per physical line, the way a ``grep`` pipeline
works, against **25** after flattening.  That is the undercount D-04's research
fell into (it recorded 12 versus 22), and it lands in exactly the region the
decision cares about, because the single-system 16-orbital masks at
``_h0ands.py:283-303`` are among the wrapped ones.  It is also why the plan's
acceptance criterion demands at least 20 sites in ``_h0ands.py``.

An honest caveat, recorded rather than implied: this module's own patterns
separate their parts with ``\\s*``, which already crosses newlines, so for
*these* patterns matching the whole file is equivalent to flattening it -- 157
records either way, 52 in ``_h0ands.py`` either way.  The flattening is kept
because it makes the guarantee structural rather than dependent on every future
pattern author remembering to use ``\\s*`` instead of a literal space, and
because it is what makes a matched snippet single-line and therefore usable as
an inventory key.  ``test_flattening_is_load_bearing`` measures both numbers on
the live file and fails if the file is ever reformatted so that nothing wraps.

Comments and string literals are blanked before matching
--------------------------------------------------------
A docstring that says ``n_orb == 16`` and an exception message that says
``n_orb == 16`` decide nothing; only executable code does.  Including prose
would force rows whose only honest disposition is "this is a sentence", so the
sweep tokenises each file and blanks every ``COMMENT`` and ``STRING`` token
before matching, preserving byte offsets so line numbers stay exact.
:func:`count_prose_mentions` reports what was excluded, so the exclusion is a
measured number in the inventory rather than an unstated choice.

The three families
------------------
``orbital-count``
    An orbital-count identifier compared against 1, 4, 9 or 16 -- the s / sp /
    spd / spdf basis sizes.  This is the family decision D-04 names.

``shell-count``
    A shell-count identifier compared against 1, 2, 3 or 4 -- s / p / d / f
    angular momentum.  This family is **not** in the plan's identifier list and
    is included deliberately.  The Phase 4 shell-resolved Coulomb defect
    (``FShellResolvedCoulombUnsupportedError``) lived at
    ``_coulomb_matrix.py:816-824``, which tests ``max_ang`` and never mentions
    ``n_orb``.  A sweep restricted to orbital-count names would have missed the
    exact defect this project already had to fix, so restricting it would be
    knowingly building a blind spot.

``basis-layout-literal``
    A bracketed integer literal that spells one of the canonical per-shell
    dimension or AO-offset tables (see :data:`BASIS_LAYOUT_SEQUENCES`).  These
    are the hardcoded-offset sites requirement CLN-03 names by hand:
    ``Structure.SHELL_DIMS`` and ``Structure.SHELL_LOCAL_STARTS`` both already
    span f, whereas ``[0, 1, 3, 5]`` stops at d and is a real gap.

Known limitation, stated rather than implied
--------------------------------------------
The ``basis-layout-literal`` family matches an enumerated set of literal
spellings.  A future phase that writes a *new* layout table with a spelling not
in :data:`BASIS_LAYOUT_SEQUENCES` will not be swept.  The enumeration is
deliberately narrow because the obvious generalisation -- "any bracketed run of
small integers" -- matches ``permute(1, 0, 2, 3)`` and dozens of other unrelated
expressions, and a row that cannot carry an honest disposition is worse than no
row.  Add the spelling here when such a table appears.

Usage
-----
::

    uv run python -c "import sys; sys.path.insert(0, 'tests'); \
from orbital_count_sweep import sweep_orbital_count_sites; \
print(len(sweep_orbital_count_sites('src/dftorch')))"

The module is named so that pytest does not collect it as a test module.
"""

from __future__ import annotations

import bisect
import io
import os
import re
import tokenize
from dataclasses import dataclass


# --------------------------------------------------------------------------
# What counts as a site
# --------------------------------------------------------------------------

#: Identifiers that hold a number of atomic orbitals (basis functions).
#: ``counts`` is included because it is the local name all three existing
#: ``_require_*`` guards bind ``n_orb[TYPE]`` to before comparing it against 16;
#: without it the guards themselves would be invisible to their own audit.
ORBITAL_COUNT_NAMES: tuple[str, ...] = (
    "n_orbitals_per_atom",
    "n_orb_per_shell",
    "n_orb_I",
    "n_orb_J",
    "n_orb",
    "norb_I",
    "norb_J",
    "norb",
    "counts",
    "nI",
    "nJ",
)

#: The four supported basis sizes: s, sp, spd, spdf.
ORBITAL_COUNT_VALUES: tuple[int, ...] = (16, 9, 4, 1)

#: Identifiers that hold a number of shells or a maximum angular momentum.
SHELL_COUNT_NAMES: tuple[str, ...] = (
    "max_ang_I",
    "max_ang_J",
    "max_ang",
    "n_shells_per_atom",
    "n_shells",
    "shell_types",
)

#: s, p, d, f expressed as shell counts / angular momentum indices.
SHELL_COUNT_VALUES: tuple[int, ...] = (4, 3, 2, 1)

#: Canonical per-shell dimension and AO-offset tables, including their
#: d-truncated forms.  ``(1, 3, 5, 7)`` is the full set of shell dimensions;
#: ``(0, 1, 4, 9)`` the AO start offsets; ``(4, 5, 6, 7, 8)`` the five d AO
#: offsets; ``(9, ..., 15)`` the seven f AO offsets; ``(1, 2, 3, 4)`` the shell
#: type ids.  A spelling that stops one entry short of f is the gap this family
#: exists to find.
BASIS_LAYOUT_SEQUENCES: tuple[tuple[int, ...], ...] = (
    (1, 3, 5),
    (1, 3, 5, 7),
    (0, 1, 3, 5),
    (0, 1, 3, 5, 7),
    (0, 1, 4),
    (0, 1, 4, 9),
    (1, 2, 3, 4),
    (4, 5, 6, 7, 8),
    (9, 10, 11, 12, 13, 14, 15),
)

FAMILY_ORBITAL_COUNT = "orbital-count"
FAMILY_SHELL_COUNT = "shell-count"
FAMILY_BASIS_LAYOUT = "basis-layout-literal"

#: Order the inventory groups its sections in.
FAMILIES: tuple[str, ...] = (
    FAMILY_ORBITAL_COUNT,
    FAMILY_SHELL_COUNT,
    FAMILY_BASIS_LAYOUT,
)


def _comparison_pattern(names: tuple[str, ...], values: tuple[int, ...]) -> str:
    dotted = r"(?:[A-Za-z_][A-Za-z_0-9]*\s*\.\s*)*"
    subscript = r"(?:\s*\[(?:[^\[\]]|\[[^\[\]]*\])*\])?"
    comparison = r"\s*(?:==|!=|>=|<=|>|<)\s*"
    name_alt = "|".join(re.escape(n) for n in names)
    value_alt = "|".join(str(v) for v in values)
    return (
        r"(?<![A-Za-z_0-9.])"
        + dotted
        + r"(?:"
        + name_alt
        + r")(?![A-Za-z_0-9])"
        + subscript
        + comparison
        + r"(?:"
        + value_alt
        + r")(?![0-9.])"
    )


def _sequence_pattern(sequences: tuple[tuple[int, ...], ...]) -> str:
    alternatives = []
    for seq in sequences:
        body = r"\s*,\s*".join(str(v) for v in seq)
        alternatives.append(r"[\[(]\s*" + body + r"\s*,?\s*[\])]")
    return "|".join(alternatives)


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        FAMILY_ORBITAL_COUNT,
        re.compile(_comparison_pattern(ORBITAL_COUNT_NAMES, ORBITAL_COUNT_VALUES)),
    ),
    (
        FAMILY_SHELL_COUNT,
        re.compile(_comparison_pattern(SHELL_COUNT_NAMES, SHELL_COUNT_VALUES)),
    ),
    (
        FAMILY_BASIS_LAYOUT,
        re.compile(_sequence_pattern(BASIS_LAYOUT_SEQUENCES)),
    ),
)


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class OrbitalCountSite:
    """One hardcoded orbital-count / basis-layout site.

    ``path`` is POSIX and relative to the swept root's parent-of-package, so it
    reads the same on Windows and Linux and can be pasted into an inventory row
    unchanged.  ``snippet`` is the matched text with all internal whitespace
    collapsed to single spaces, which makes it a stable key: two sites on the
    same physical line (``(nI == 1) & (nJ == 1)``) differ in their snippet.
    """

    path: str
    line: int
    snippet: str
    family: str

    @property
    def key(self) -> tuple[str, int, str]:
        """The identity used to match a swept site against an inventory row."""
        return (self.path, self.line, self.snippet)


# --------------------------------------------------------------------------
# Source preparation
# --------------------------------------------------------------------------


def _prose_token_types() -> frozenset[int]:
    """Token types whose contents are prose rather than executable code.

    ``FSTRING_START`` / ``FSTRING_MIDDLE`` / ``FSTRING_END`` exist only from
    Python 3.12, where f-strings stopped being a single ``STRING`` token.  They
    are looked up rather than referenced so this module keeps working on 3.11,
    and they are genuinely required on 3.12+: without them every ``n_orb == 16``
    written inside a guard's own f-string error message is swept as if it were a
    routing decision, and the guards' real ``counts == 16`` tests are the ones
    that go missing.  Measured on 3.13 before this was fixed.
    """
    types = {tokenize.COMMENT, tokenize.STRING}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        value = getattr(tokenize, name, None)
        if value is not None:
            types.add(value)
    return frozenset(types)


_PROSE_TOKEN_TYPES: frozenset[int] = _prose_token_types()


def blank_comments_and_strings(text: str) -> str:
    """Return ``text`` with every comment and string literal turned into spaces.

    Byte offsets and newlines are preserved exactly, so a line number recovered
    from the blanked text is the line number in the original file.  If the file
    cannot be tokenised (a syntax error, an exotic encoding) the original text
    is returned unchanged: over-reporting a prose mention is a visible, fixable
    inventory row, whereas under-reporting is the failure this module exists to
    prevent.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text

    line_starts = [0]
    for line in text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    chars = list(text)
    for token in tokens:
        if token.type not in _PROSE_TOKEN_TYPES:
            continue
        start = line_starts[token.start[0] - 1] + token.start[1]
        end = line_starts[token.end[0] - 1] + token.end[1]
        for i in range(start, min(end, len(chars))):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def _flatten_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    """Collapse each newline-plus-indent run to one space.

    Returns ``(flat, flat_starts, orig_starts)``.  ``flat_starts`` and
    ``orig_starts`` are parallel ascending lists describing the segments that
    survived unchanged, so a flat offset can be mapped back to an original
    offset by bisection.  Offsets that land on an inserted space map to the
    start of the collapsed run, which is the newline -- i.e. the end of the
    preceding line.  That case cannot begin a match, because every pattern in
    this module starts with a non-space character.
    """
    pieces: list[str] = []
    flat_starts: list[int] = []
    orig_starts: list[int] = []
    flat_len = 0
    pos = 0

    for match in re.finditer(r"\n[ \t]*", text):
        segment = text[pos : match.start()]
        if segment:
            pieces.append(segment)
            flat_starts.append(flat_len)
            orig_starts.append(pos)
            flat_len += len(segment)
        pieces.append(" ")
        flat_starts.append(flat_len)
        orig_starts.append(match.start())
        flat_len += 1
        pos = match.end()

    tail = text[pos:]
    if tail:
        pieces.append(tail)
        flat_starts.append(flat_len)
        orig_starts.append(pos)

    return "".join(pieces), flat_starts, orig_starts


def _to_original_offset(
    flat_pos: int, flat_starts: list[int], orig_starts: list[int]
) -> int:
    index = bisect.bisect_right(flat_starts, flat_pos) - 1
    if index < 0:
        return 0
    return orig_starts[index] + (flat_pos - flat_starts[index])


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------


def iter_source_files(root: str | os.PathLike[str]):
    """Yield every ``.py`` file under ``root``, ``__pycache__`` excluded.

    ``_legacy`` and ``sedacs`` are **not** excluded.  Whether a module is
    reachable is a disposition the inventory records with evidence, never a
    reason to leave it out of the sweep -- the plan-time research omitted
    ``_legacy/H0andS.py`` entirely and thereby missed the largest single
    concentration of 1/4/9 masks in the package.
    """
    root = os.fspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _relative_posix(path: str, root: str) -> str:
    root = os.fspath(root)
    parent = os.path.dirname(os.path.abspath(root)) or os.path.abspath(root)
    return os.path.relpath(os.path.abspath(path), parent).replace(os.sep, "/")


def sweep_orbital_count_sites(
    root: str | os.PathLike[str] = "src/dftorch",
) -> list[OrbitalCountSite]:
    """Return every hardcoded orbital-count / basis-layout site under ``root``.

    Records are sorted by ``(path, line, snippet, family)`` so the ordering is
    stable across runs and platforms.  A site is reported once per match, so a
    line carrying two comparisons yields two records; the inventory carries one
    row each.
    """
    sites: list[OrbitalCountSite] = []
    for path in iter_source_files(root):
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        code = blank_comments_and_strings(text)
        flat, flat_starts, orig_starts = _flatten_with_offsets(code)
        relative = _relative_posix(path, root)
        for family, pattern in _PATTERNS:
            for match in pattern.finditer(flat):
                offset = _to_original_offset(match.start(), flat_starts, orig_starts)
                line = text.count("\n", 0, offset) + 1
                snippet = " ".join(match.group(0).split())
                sites.append(OrbitalCountSite(relative, line, snippet, family))
    return sorted(sites)


def count_prose_mentions(root: str | os.PathLike[str] = "src/dftorch") -> int:
    """Count pattern matches that live in a comment or a string literal.

    These are excluded from :func:`sweep_orbital_count_sites` because prose
    decides nothing.  The number is reported in the inventory so the exclusion
    is measured rather than merely asserted.
    """
    total = 0
    for path in iter_source_files(root):
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        code = blank_comments_and_strings(text)
        for _family, pattern in _PATTERNS:
            everything = len(pattern.findall(re.sub(r"\n[ \t]*", " ", text)))
            executable = len(pattern.findall(re.sub(r"\n[ \t]*", " ", code)))
            total += everything - executable
    return total


def flatten_source(text: str) -> str:
    """Collapse each newline-plus-indent run in ``text`` to a single space.

    Exposed so a test can demonstrate that the flattening is load-bearing:
    reflow a real comparison across lines, and the same pattern finds strictly
    fewer matches without it.
    """
    return _flatten_with_offsets(text)[0]


def pattern_for(family: str) -> re.Pattern[str]:
    """Return the compiled pattern for one of :data:`FAMILIES`."""
    for name, pattern in _PATTERNS:
        if name == family:
            return pattern
    raise KeyError(f"unknown family {family!r}; known: {FAMILIES}")


def count_by_file(sites: list[OrbitalCountSite]) -> dict[str, int]:
    """Sites per file, for the inventory's summary table."""
    counts: dict[str, int] = {}
    for site in sites:
        counts[site.path] = counts.get(site.path, 0) + 1
    return counts


if __name__ == "__main__":  # pragma: no cover - operator convenience
    found = sweep_orbital_count_sites("src/dftorch")
    for name, count in sorted(count_by_file(found).items()):
        print(f"{count:5d}  {name}")
    print(f"{len(found):5d}  TOTAL")
