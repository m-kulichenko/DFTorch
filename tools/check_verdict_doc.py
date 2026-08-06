"""Completeness and citation gate for docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md.

Standard library only.  Run from anywhere::

    python tools/check_verdict_doc.py

Exits 0 when the verdict document is complete and its source citations still
point at what it says they point at; exits 1 with a numbered list of problems
otherwise.

What this checks, and what it deliberately does not
---------------------------------------------------
This is a *completeness and citation* gate, not a physics check.  It cannot
tell whether the verdict is correct.  It can tell whether the document still
states a verdict, still carries the three-way split that decision D-6.03
requires, still decodes as ASCII, still keeps test code out, and - the one that
earns this script its existence - whether every source line the document cites
still contains what the document claims.

That last check is the point.  A document that describes code is trusted, so a
document that has drifted out of agreement with the code is worse than no
document.  Plans 06-01 through 06-03 already moved both of the cited call sites
once; without a gate, the next such move would leave this document quietly
wrong while still reading as authoritative.

Physics claims are held by tests/test_energy_definitions_f.py instead.
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "SINGLE-SHOT-VS-SELF-CONSISTENT.md"

# --- Required prose anchors -------------------------------------------------
#
# Each entry is (substring, why it has to be there).  The "why" is printed on
# failure so that whoever broke it learns what the anchor was protecting rather
# than just which string vanished.
REQUIRED_ANCHORS = [
    (
        "definition, not a defect",
        "decision D-6.03 requires a stated verdict, and one of the two answers "
        "it allows; a document that concludes nothing is not a verdict",
    ),
    (
        "84.2 %",
        "decision D-6.03 rejects an explanation that only accounts for electron "
        "repulsion; the measured share sitting in the band-structure term has "
        "to be stated as a number",
    ),
    (
        "84 percent",
        "the same share has to be stated in words as well as in the table, "
        "because a reader who skims the table can still miss the point",
    ),
    (
        "band-structure energy",
        "the three-way split needs a band-structure row",
    ),
    (
        "electron-repulsion energy",
        "the three-way split needs an electron-repulsion row",
    ),
    (
        "EU_N_REFERENCE_E_TOT",
        "the Phase 4 pinned reference has to be named, since a reader applying "
        "it to the settled path is the confusion this document exists to stop",
    ),
    (
        "one-pass path only",
        "the pin has to be stated as belonging to the one-pass path only",
    ),
    (
        "D-11",
        "the Phase 4 decision that created the one-pass definition has to be "
        "named, so a reader can go and read it",
    ),
    (
        "2026-08-04",
        "the dissociation-charge residual is a closed human ruling of that "
        "date, and has to be recorded as closed rather than as an open item",
    ),
    (
        "accepted",
        "the same ruling accepted the residual; 'accepted' is the word that "
        "makes it closed",
    ),
    (
        "D-6.08",
        "the document must say that no settled value is validated or frozen",
    ),
]

# The verdict has to come first, not be buried.  Everything before this heading
# counts as "the first section".
FIRST_SECTION_END = "## Vocabulary"
VERDICT_ANCHOR = "definition, not a defect"

# --- Required source citations ----------------------------------------------
#
# (path relative to repo root, first line, last line, text that must appear
# somewhere in that line range, what the citation is for).
#
# Line numbers are 1-based and inclusive, matching how the document writes them.
REQUIRED_CITATIONS = [
    (
        "src/dftorch/ESDriver.py",
        1030,
        1031,
        "None,  # C: no Coulomb matrix",
        "the two None arguments that switch the electron-repulsion term off on "
        "the one-pass path - the whole mechanism the verdict rests on",
    ),
    (
        "src/dftorch/ESDriver.py",
        1018,
        1041,
        ") = energy(",
        "the single-shot call to energy() that those None arguments belong to",
    ),
    (
        "src/dftorch/ESDriver.py",
        958,
        973,
        "D-11",
        "the comment that states the reason for the zero in the code itself",
    ),
    (
        "src/dftorch/_energy.py",
        181,
        182,
        "Ecoul = 0",
        "the arm of energy()'s four-way branch that the two Nones select",
    ),
    (
        "src/dftorch/Constants.py",
        232,
        232,
        "self.U = torch.nn.Parameter(US",
        "the per-atom Hubbard strength assignment whose recorded defect note "
        "this document cross-references",
    ),
]

# --- Things that must NOT be in the document --------------------------------
#
# The verdict quotes measurements as prose evidence.  A measurement that turns
# into an assertion becomes a frozen reference value around behaviour this
# phase is still changing, which decision D-6.08 forbids.
FORBIDDEN_LINE_PREFIXES = [
    ("assert ", "an assertion - measurements belong here as prose, not as a test"),
    ("def test_", "test code - the tests live in tests/test_energy_definitions_f.py"),
    ("pytest", "test-runner invocation - this is a document, not a test module"),
]


def _read_lines(path):
    """Return the file's lines, or None if it is not there."""
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None


def check():
    """Return a list of problem descriptions; empty means the document passes."""
    problems = []

    raw = None
    try:
        raw = DOC_PATH.read_bytes()
    except OSError:
        problems.append(
            f"{DOC_PATH} does not exist. The verdict document is the whole "
            "deliverable of plan 06-04 task 1."
        )
        return problems

    if raw and max(raw) > 127:
        offending = [
            (index, byte)
            for index, byte in enumerate(raw)
            if byte > 127
        ]
        first_index, first_byte = offending[0]
        line_number = raw[:first_index].count(b"\n") + 1
        problems.append(
            f"{DOC_PATH.name} is not pure ASCII: byte {first_byte} at line "
            f"{line_number} ({len(offending)} non-ASCII bytes in total). "
            "A non-ASCII character renders as a replacement box on a Windows "
            "cp1252 console. Write 'A' for Angstrom and '->' for an arrow."
        )

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # 1. Prose anchors.
    for anchor, why in REQUIRED_ANCHORS:
        if anchor not in text:
            problems.append(
                f"missing required text {anchor!r} - {why}"
            )

    # 2. The verdict has to be in the first section, not buried later.
    if VERDICT_ANCHOR in text:
        cut = text.find(FIRST_SECTION_END)
        if cut == -1:
            problems.append(
                f"heading {FIRST_SECTION_END!r} is gone, so this script cannot "
                "tell where the first section ends; restore it or update this "
                "checker deliberately"
            )
        elif text.find(VERDICT_ANCHOR) > cut:
            problems.append(
                f"the verdict ({VERDICT_ANCHOR!r}) appears only after "
                f"{FIRST_SECTION_END!r}. D-6.03 wants the verdict stated first, "
                "not reached at the end of an argument."
            )

    # 3. Forbidden content.
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        for prefix, why in FORBIDDEN_LINE_PREFIXES:
            if stripped.startswith(prefix):
                problems.append(
                    f"line {number} starts with {prefix!r}, which reads as "
                    f"{why}. The line is: {stripped!r}"
                )

    # 4. Source citations - both that the document makes them, and that the
    #    cited lines still contain what it says they contain.
    for rel_path, start, end, needle, why in REQUIRED_CITATIONS:
        file_name = rel_path.rsplit("/", 1)[-1]
        citation = (
            f"{file_name}:{start}" if start == end else f"{file_name}:{start}-{end}"
        )
        if citation not in text:
            problems.append(
                f"the document does not cite {citation} - {why}"
            )

        source_lines = _read_lines(REPO_ROOT / rel_path)
        if source_lines is None:
            problems.append(
                f"cited source file {rel_path} does not exist, so citation "
                f"{citation} cannot mean anything"
            )
            continue
        if end > len(source_lines):
            problems.append(
                f"citation {citation} runs past the end of {rel_path}, which "
                f"has only {len(source_lines)} lines. The code moved; find the "
                f"new location of ({needle!r}) and update both the document "
                "and this script."
            )
            continue
        window = "\n".join(source_lines[start - 1 : end])
        if needle not in window:
            problems.append(
                f"citation {citation} no longer contains {needle!r} - {why}. "
                f"{rel_path} lines {start}-{end} now read:\n"
                + "\n".join(
                    f"    {n}: {source_lines[n - 1]}"
                    for n in range(start, min(end, start + 6) + 1)
                )
                + "\n  The document has drifted out of agreement with the code "
                "it describes. Re-confirm the line numbers, update the "
                "document, and update this script's citation table."
            )

    return problems


def main():
    problems = check()
    if problems:
        print(f"FAIL: {DOC_PATH.name} has {len(problems)} problem(s).")
        for number, problem in enumerate(problems, start=1):
            print(f"  {number}. {problem}")
        return 1
    print(
        f"OK: {DOC_PATH.name} states a verdict, carries the three-way split, "
        f"is pure ASCII, contains no test code, and all "
        f"{len(REQUIRED_CITATIONS)} source citations still point at what it "
        "says they point at."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
