import json

from scripts.release.scan_anonymous_supplement import scan_tree


def test_scanner_flags_identity_urls_authors_and_notebook_outputs(tmp_path):
    target = tmp_path / "anonymous_supplement"
    target.mkdir()
    (target / "README.md").write_text(
        "See AUTHORS.md and https://github.com/example-user/project.\n",
        encoding="utf-8",
    )
    (target / "results.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "outputs": [
                            {
                                "output_type": "stream",
                                "text": ["Ada Lovelace\n"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    findings = scan_tree(
        target,
        {
            "names": {"Ada Lovelace"},
            "emails": set(),
            "usernames": {"ada-lovelace"},
        },
    )
    categories = {finding.category for finding in findings}
    assert "author_name" in categories
    assert "authors_reference" in categories
    assert "github_username" in categories
    assert "notebook_output_identity" in categories
