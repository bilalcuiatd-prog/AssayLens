FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    ASSAYLENS_DATA_DIR=/tmp/assaylens-data

WORKDIR /srv/assaylens
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 assay
COPY --chown=assay:assay . .
RUN python scripts/verify_deploy.py
USER assay
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn render_app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
