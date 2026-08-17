#!/bin/sh
# Render deploy target entrypoint. Mirrors backend/docker-entrypoint.sh's
# intent (migrate, then serve) but for the ONE combined process
# (render_app.py) instead of Django alone -- see DEPLOY_NOTES.md.
set -e

# RISK_API_URL is computed here, not baked in at build time, because $PORT
# is only known at container start (Render assigns it dynamically). This is
# what lets backend/users/services/prediction_service.py's EXISTING
# HTTP-client code work completely unchanged on Render: it just calls
# itself over loopback, at the FastAPI mount point (/api), instead of a
# separate api container.
export RISK_API_URL="http://127.0.0.1:${PORT:-8000}/api"

cd backend
python manage.py migrate --noinput
cd ..

# Idempotent -- skips entirely if `decisions` already has >= 250 rows (see
# render_seed.py's own docstring). Runs every container start on purpose:
# the cheap, correct way to guarantee a fresh Neon database gets seeded on
# first boot without a separate one-off Render job.
python render_seed.py

exec uvicorn render_app:app --host 0.0.0.0 --port "${PORT:-8000}"
