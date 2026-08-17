# Screenshots

Captured 2026-08-17 against the live docker-compose stack.

- `monitoring_dashboard.png`, `reliability_curve.png`, `sandbox_form.png`,
  `api_docs.png` — current.
- `review_queue.png` — **predates the `ml/src/policy/reason_codes.py` NaN
  fix** (see PROGRESS.md's Phase 7 section). A few rows show a literal
  `"nan days since this payer last paid this payee"` in their reason codes —
  that bug is fixed in code, but re-capturing this screenshot requires
  re-scoring events through the live worker (Docker), which was unavailable
  when the fix landed. Pending re-capture next time the stack is up; not
  embedded in README.md in the meantime.
