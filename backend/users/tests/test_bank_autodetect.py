"""Regression test for the sandbox form's client-side VPA-suffix-to-bank
auto-detect (app/model.html's inline <script>). No JS test runner exists in
this repo -- rather than add one for a single small mapping, this extracts
the actual `BANK_SUFFIXES` object literal straight out of the template file
and evaluates it as JSON, so a future edit to the real mapping is what this
test checks, not a hand-copied duplicate that could silently drift from it.

Found via a live bug report (2026-08-19): the sandbox showed OTHER for
alice@okaxis and merchant@ybl. Both suffixes were already correctly mapped
in the code at the time -- the actual fix was triggering the derivation on
'input' as well as 'blur' (a user who never left the VPA field before
submitting saw a stale default), plus expanding coverage to the other
common handles named in the bug report. This test locks in the mapping
values; it can't exercise the event-wiring fix itself without a browser.

Uses django.test.SimpleTestCase (no DB access needed) -- same convention as
test_evaluation.py, and the one that makes this actually run under
`manage.py test users` (CI's own invocation), unlike a bare pytest-style
function, which Python's unittest-based test discovery never sees.
"""
import json
import re
from pathlib import Path

from django.test import SimpleTestCase

TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3] / "backend" / "users" / "templates" / "app" / "model.html"
)


def _load_bank_suffixes() -> dict:
    text = TEMPLATE_PATH.read_text()
    match = re.search(r"const BANK_SUFFIXES = (\{.*?\});", text, re.DOTALL)
    assert match, "BANK_SUFFIXES object literal not found in app/model.html"
    # The literal uses single-quoted JS strings -- valid JSON needs double
    # quotes. Safe here because none of the keys/values contain a quote
    # character themselves (they're bare bank-suffix/code strings).
    js_literal = match.group(1)
    json_literal = js_literal.replace("'", '"')
    json_literal = re.sub(r",(\s*})", r"\1", json_literal)  # trailing comma before a closing brace
    return json.loads(json_literal)


class BankAutoDetectTests(SimpleTestCase):
    def test_bank_suffixes_covers_the_common_handles(self):
        suffixes = _load_bank_suffixes()

        expected = {
            "okaxis": "AXIS", "okicici": "ICICI", "oksbi": "SBI", "okhdfcbank": "HDFC",
            "ybl": "YES", "ibl": "IDBI", "axl": "AXIS",
            "paytm": "PAYTM", "upi": "NPCI", "apl": "AXIS", "jupiteraxis": "AXIS",
            "fbl": "FEDERAL", "sbi": "SBI", "hdfcbank": "HDFC", "icici": "ICICI", "kotak": "KOTAK",
        }
        for suffix, bank in expected.items():
            self.assertIn(suffix, suffixes, f"{suffix!r} missing from BANK_SUFFIXES")
            self.assertEqual(suffixes[suffix], bank, f"{suffix!r} -> {suffixes[suffix]!r}, expected {bank!r}")

    def test_the_two_suffixes_from_the_original_bug_report(self):
        """alice@okaxis and merchant@ybl specifically -- the exact report."""
        suffixes = _load_bank_suffixes()
        self.assertEqual(suffixes["okaxis"], "AXIS")
        self.assertEqual(suffixes["ybl"], "YES")
