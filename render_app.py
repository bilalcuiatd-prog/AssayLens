from __future__ import annotations

import base64
import binascii
import hmac
import os

from fastapi.responses import JSONResponse, PlainTextResponse
from assaylens.studio_api import app as research_app


class ResearchAccessGate:
    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.application(scope, receive, send)

        if scope.get("path") == "/healthz":
            return await JSONResponse({"status": "ok"})(scope, receive, send)

        username = os.environ.get("ASSAYLENS_SITE_USERNAME", "")
        password = os.environ.get("ASSAYLENS_SITE_PASSWORD", "")
        if not username or not password:
            return await PlainTextResponse(
                "Access credentials are not configured.", status_code=503
            )(scope, receive, send)

        headers = dict(scope.get("headers", ()))
        authorization = headers.get(b"authorization", b"")
        if authorization.startswith(b"Basic "):
            try:
                supplied = base64.b64decode(
                    authorization[6:], validate=True
                ).decode("utf-8")
                received_user, separator, received_password = supplied.partition(":")
                if (
                    separator
                    and hmac.compare_digest(received_user, username)
                    and hmac.compare_digest(received_password, password)
                ):
                    return await self.application(scope, receive, send)
            except (binascii.Error, UnicodeError, ValueError):
                pass

        return await PlainTextResponse(
            "Sign-in required.",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="AssayLens Research"'},
        )(scope, receive, send)


app = ResearchAccessGate(research_app)
