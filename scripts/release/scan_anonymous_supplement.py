#!/usr/bin/env python3
"""Scan a future anonymous supplement for author-identifying material."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


URL_RE = re.compile(r"https?://[^\s)\]>\x27\x22]+", re.IGNORECASE)
AUTHORS_REF_RE = re.compile(r"\bAUTHORS(?:\.md)?\b", re.IGNORECASE)
GITHUB_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/([^/\s?#]+)", re.IGNORECASE
)
HF_URL_RE = re.compile(
    r"https?://(?:www\.)?huggingface\.co/([^/\s?#]+)", re.IGNORECASE
)


@dataclass(frozen=True)
class Finding:
    category: str
    path: str
    line: int
    match: str
    detail: str


def _repo_authors(repo_root: Path) -> dict[str, set[str]]:
    names: set[str] = set()
    emails: set[str] = set()
    authors_path = repo_root / "AUTHORS.md"
    if authors_path.is_file():
        for line in authors_path.read_text(encoding="utf-8").splitlines():
            match = re.match(
                r"\s*-\s*(.+?)\s+[—-]\s+\x60([^\x60]+)\x60", line
            )
            if match:
                names.add(match.group(1).strip())
                emails.add(match.group(2).strip())
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "log", "--format=%aN\t%aE"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        output = ""
    for line in output.splitlines():
        name, _, email = line.partition("\t")
        if name and name.casefold() not in {"codex", "unknown"}:
            names.add(name.strip())
        if email:
            emails.add(email.strip())
    usernames = set()
    for email in emails:
        local = email.split("@", 1)[0]
        if "+" in local:
            usernames.add(local.split("+", 1)[1])
    return {"names": names, "emails": emails, "usernames": usernames}


def _identity_patterns(
    identity: dict[str, set[str]],
) -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for name in sorted(identity.get("names", set()), key=len, reverse=True):
        if len(name) >= 4:
            patterns.append(
                (
                    "author_name",
                    re.compile(
                        r"(?<!\w){}(?!\w)".format(re.escape(name)), re.IGNORECASE
                    ),
                )
            )
    for email in sorted(identity.get("emails", set()), key=len, reverse=True):
        patterns.append(
            ("author_email", re.compile(re.escape(email), re.IGNORECASE))
        )
    for username in sorted(identity.get("usernames", set()), key=len, reverse=True):
        if len(username) >= 3:
            patterns.append(
                (
                    "github_username",
                    re.compile(
                        r"(?<![\w-]){}(?![\w-])".format(re.escape(username)),
                        re.IGNORECASE,
                    ),
                )
            )
    return patterns


def _line_findings(
    path: Path,
    line_number: int,
    text: str,
    identity: dict[str, set[str]],
    *,
    in_notebook_output: bool = False,
) -> list[Finding]:
    findings: list[Finding] = []
    relative = str(path)
    for category, pattern in _identity_patterns(identity):
        for match in pattern.finditer(text):
            findings.append(
                Finding(
                    category,
                    relative,
                    line_number,
                    match.group(0),
                    "known repository identity",
                )
            )
    for match in AUTHORS_REF_RE.finditer(text):
        findings.append(
            Finding(
                "authors_reference",
                relative,
                line_number,
                match.group(0),
                "AUTHORS.md reference",
            )
        )
    for match in GITHUB_URL_RE.finditer(text):
        findings.append(
            Finding(
                "github_username",
                relative,
                line_number,
                match.group(0),
                "GitHub identity URL",
            )
        )
    for match in HF_URL_RE.finditer(text):
        findings.append(
            Finding(
                "huggingface_username",
                relative,
                line_number,
                match.group(0),
                "Hugging Face identity URL",
            )
        )
    for match in URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:")
        findings.append(
            Finding(
                "external_url_review",
                relative,
                line_number,
                url,
                "external URL requires anonymization review",
            )
        )
        if "github.com" in url.casefold() or "huggingface.co" in url.casefold():
            category = (
                "documentation_deanonymizing_link"
                if path.suffix.casefold() in {".md", ".rst", ".html", ".ipynb"}
                else "identifying_url"
            )
            findings.append(
                Finding(
                    category,
                    relative,
                    line_number,
                    url,
                    "identity-bearing external URL",
                )
            )
    if in_notebook_output and findings:
        findings.append(
            Finding(
                "notebook_output_identity",
                relative,
                line_number,
                text.strip()[:240],
                "identity material in notebook output",
            )
        )
    return findings


def _scan_notebook(path: Path, identity: dict[str, set[str]]) -> list[Finding]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    findings: list[Finding] = []

    def string_values(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from string_values(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from string_values(item)

    for index, cell in enumerate(payload.get("cells", [])):
        for output in cell.get("outputs", []):
            values: list[str] = []
            for key in ("text", "data"):
                values.extend(string_values(output.get(key)))
            if isinstance(output.get("ename"), str):
                values.append(output["ename"])
            for value in values:
                findings.extend(
                    _line_findings(
                        path,
                        index + 1,
                        value,
                        identity,
                        in_notebook_output=True,
                    )
                )
    return findings


def scan_tree(target: Path, identity: dict[str, set[str]]) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(p for p in target.rglob("*") if p.is_file()):
        if path.suffix.casefold() == ".ipynb":
            findings.extend(_scan_notebook(path, identity))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            findings.extend(_line_findings(path, line_number, line, identity))
    return findings


def scan(repo_root: Path, target: Path) -> dict[str, Any]:
    identity = _repo_authors(repo_root)
    if not target.exists():
        return {
            "status": "NOT_READY_TARGET_MISSING",
            "target": str(target),
            "finding_count": 0,
            "findings": [],
            "identity_sources": {
                key: sorted(value) for key, value in identity.items()
            },
        }
    findings = scan_tree(target, identity)
    return {
        "status": "PASS" if not findings else "FAIL",
        "target": str(target),
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
        "identity_sources": {
            key: sorted(value) for key, value in identity.items()
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--target", type=Path, default=Path("release/anonymous_supplement")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/revision/t8_checks/anonymization_scan.json"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    target = (repo_root / args.target).resolve()
    result = scan(repo_root, target)
    output = (repo_root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("{}: {}".format(result["status"], output))
    return 0 if result["status"] in {"PASS", "NOT_READY_TARGET_MISSING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
