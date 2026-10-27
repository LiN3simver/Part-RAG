"""Signed anonymous browser identity for native Part-RAG conversations, not SSO."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Response

COOKIE = "partrag_session"
MAX_AGE = 30 * 24 * 3600


def _secret() -> bytes:
    secret = os.getenv("RAG_SESSION_SECRET", "")
    if len(secret) < 32:
        raise HTTPException(503, "原生對話未啟用：請設定至少 32 字元的 RAG_SESSION_SECRET")
    return secret.encode("utf-8")


def sign(session_id: str, issued: int, secret: bytes) -> str:
    payload = f"{session_id}.{issued}"
    signature = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def verify(cookie: str, secret: bytes, now: int | None = None) -> str | None:
    if len(cookie) > 160:
        return None
    try:
        sid, issued_text, signature = cookie.split(".")
        issued = int(issued_text)
    except (ValueError, TypeError):
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", sid) or not re.fullmatch(r"[0-9a-f]{64}", signature):
        return None
    age = (int(time.time()) if now is None else now) - issued
    if age < 0 or age > MAX_AGE:
        return None
    expected = sign(sid, issued, secret).rsplit(".", 1)[1]
    return sid if hmac.compare_digest(expected, signature) else None


def stamp(request: Request, response: Response) -> None:
    token = getattr(request.state, "new_rag_session", None)
    if token:
        secure = (os.getenv("RAG_SESSION_SECURE", "").lower() in {"1", "true"}
                  or request.url.scheme == "https")
        response.set_cookie(COOKIE, token, max_age=MAX_AGE, httponly=True,
                            secure=secure, samesite="lax", path="/api")


def owner_id(request: Request, response: Response) -> str:
    secret = _secret()
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if request.headers.get("sec-fetch-site") == "cross-site" or (
            origin and (urlsplit(origin).scheme, urlsplit(origin).netloc) !=
            (request.url.scheme, request.url.netloc)
        ):
            raise HTTPException(403, "不接受跨站對話修改")
    sid = verify(request.cookies.get(COOKIE, ""), secret)
    if sid is None:
        sid = secrets.token_hex(32)
        request.state.new_rag_session = sign(sid, int(time.time()), secret)
        stamp(request, response)
    elif int(time.time()) - int(request.cookies[COOKIE].split(".")[1]) > MAX_AGE // 2:
        request.state.new_rag_session = sign(sid, int(time.time()), secret)
        stamp(request, response)
    # Store a one-way identifier; the DB does not store the browser's bearer cookie.
    return hashlib.sha256(sid.encode()).hexdigest()
