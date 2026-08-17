# Demo script (90 seconds)

Shot-by-shot. Record at 1080p, no editing needed if you follow this order —
each beat is one continuous action. Have the stack already running
(`docker compose up -d`) before you hit record; don't waste seconds waiting
on health checks on camera.

| # | Time | Shot | What to show | What to say |
|---|---|---|---|---|
| 1 | 0:00–0:10 | Landing page | `localhost:8000/` — log in, land on the home page. | "This is Sentinel — real-time UPI fraud scoring. Everything you're about to see is running locally against synthetic data." |
| 2 | 0:10–0:20 | Replay running | A terminal with `docker compose --profile tools run generator --limit 2000 --speed 20` already running, log lines scrolling. | "I'm replaying synthetic transactions through the same Redis Streams pipeline a real payment app would push into." |
| 3 | 0:20–0:35 | Live scoring | `localhost:8000/monitoring/`, refresh once or twice — action mix / alert rate counts visibly ticking up. | "Each event gets scored in real time — cold or warm model depending on whether we've seen this payee before, calibrated probability, cost-based decision." |
| 4 | 0:35–0:45 | An alert appearing | `localhost:8000/review/` — scroll to a REVIEW or BLOCK row. | "Anything above the review threshold lands in the analyst queue — here's one that got flagged." |
| 5 | 0:45–0:60 | Reason codes | Zoom/crop into that row's reason-codes column. | "No hardcoded explanation text — these are SHAP values, computed per transaction, so the explanation is genuinely about *this* score." |
| 6 | 0:60–0:70 | Analyst labels it | Click the disposition dropdown, choose "Confirmed fraud" (or "Legit"), hit Save. | "An analyst reviews it and labels it — that label never touches the original decision record, it's append-only." |
| 7 | 0:70–0:82 | Monitoring updates | Back to `localhost:8000/monitoring/`, point at the review-queue-health stat row (reviewed count went up). | "That label feeds straight into the monitoring dashboard — queue depth, precision on reviewed cases, with the selective-labelling caveat right there so it's never misread as the model's true precision." |
| 8 | 0:82–0:90 | `/api/docs` | `localhost:8001/docs` — expand `POST /v1/score`. | "And the scoring service itself is a standalone FastAPI app with a real OpenAPI schema — Django's just a client of it, same as any external caller would be." |

**Before recording:**
- Run the replay once beforehand off-camera so `/monitoring/` and `/review/`
  already have a few REVIEW/BLOCK rows to scroll to at step 4 — don't rely on
  the on-camera replay alone landing a fraud typology in the first 2,000
  events.
- Log in as a demo user ahead of time (`/register/` once, then just `/login/`
  on camera) so step 1 doesn't eat time on account creation.
- Keep the terminal font large enough to read the throughput log lines in
  step 2 — that's the "look, it's actually processing a stream" beat.
