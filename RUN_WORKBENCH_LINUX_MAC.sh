#!/bin/sh
set -e
cd "$(dirname "$0")"
python3 -m uvicorn assaylens.studio_api:app --host 127.0.0.1 --port 8000
