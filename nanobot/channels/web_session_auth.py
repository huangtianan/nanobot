"""Resolve nanobot Web channel session keys from an external auth service (Bearer token → user id)."""

from __future__ import annotations

from typing import Any

from loguru import logger


def web_session_auth_settings(web_cfg: Any) -> tuple[str, str, float]:
    """Return (base_url, me_path, timeout) from channels.web config dict or WebConfig."""
    if web_cfg is None:
        return "", "/auth/users/me", 10.0
    if isinstance(web_cfg, dict):
        base = str(
            web_cfg.get("sessionAuthBaseUrl") or web_cfg.get("session_auth_base_url") or ""
        ).strip()
        path = str(
            web_cfg.get("sessionAuthMePath")
            or web_cfg.get("session_auth_me_path")
            or "/auth/users/me"
        )
        try:
            to = float(web_cfg.get("sessionAuthTimeout") or web_cfg.get("session_auth_timeout") or 10.0)
        except (TypeError, ValueError):
            to = 10.0
        return base, path, to
    base = str(getattr(web_cfg, "session_auth_base_url", "") or "").strip()
    path = str(getattr(web_cfg, "session_auth_me_path", "/auth/users/me") or "/auth/users/me")
    try:
        to = float(getattr(web_cfg, "session_auth_timeout", 10.0) or 10.0)
    except (TypeError, ValueError):
        to = 10.0
    return base, path, to


def bearer_token_from_message_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Return DataAgent / SSO bearer token from inbound web message metadata (mirrors data_agent overlay)."""
    if not metadata:
        return None
    tok = metadata.get("auth_token") or metadata.get("authToken")
    if isinstance(tok, str) and tok.strip():
        return tok.strip()
    nested = metadata.get("data_agent") or metadata.get("dataAgent")
    if isinstance(nested, dict):
        t2 = nested.get("auth_token") or nested.get("authToken")
        if isinstance(t2, str) and t2.strip():
            return t2.strip()
    return None


async def resolve_web_session_user_id(
    *,
    base_url: str,
    me_path: str,
    bearer_token: str,
    timeout: float = 10.0,
) -> str | None:
    """GET ``me_path`` with ``Authorization: Bearer``; parse ``data.id`` (or top-level id)."""

    try:
        import httpx
    except ImportError:
        logger.warning("Web session auth: httpx not installed")
        return None

    root = base_url.strip().rstrip("/")
    path = (me_path or "/auth/users/me").strip()
    if not path.startswith("/"):
        path = "/" + path
    url = f"{root}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {bearer_token.strip()}"},
            )
            resp.raise_for_status()
            body = resp.json()
    except Exception as e:
        logger.warning("Web session auth: user resolve failed: {}: {}", type(e).__name__, e)
        return None

    if not isinstance(body, dict):
        return None
    data = body.get("data", body)
    if not isinstance(data, dict):
        return None
    uid = data.get("id") or data.get("user_id") or data.get("userId")
    if uid is None:
        logger.warning("Web session auth: no user id in /me payload keys={}", list(data.keys())[:20])
        return None
    s = str(uid).strip()
    return s or None
