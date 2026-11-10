
from assaylens.studio_api import app


@app.get("/healthz")
def health_check():
    return {"status": "ok"}
