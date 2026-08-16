FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps needed to build cryptography/Pillow wheels on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x backend/docker-entrypoint.sh \
    && cd backend && DEBUG=True python manage.py collectstatic --noinput

# Shared image for ui/api/worker/generator (Phase 5) -- one build, reused by all
# four docker-compose services via per-service `command`/`working_dir` instead of
# a baked-in ENTRYPOINT, so disk isn't spent on four near-identical images.
# The Django container sets working_dir: /app/backend and
# entrypoint: ["./docker-entrypoint.sh"] itself in docker-compose.yml.
EXPOSE 8000
