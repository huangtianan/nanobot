"""Small HTTP API alongside the WebSocket channel (e.g. session reset from a browser button)."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from aiohttp import web
from loguru import logger

from nanobot.channels.web_session_auth import (
    resolve_web_session_user_id,
    web_session_auth_settings,
)

if TYPE_CHECKING:
    from nanobot.session.manager import SessionManager


def _cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type",
        "Access-Control-Max-Age": "86400",
    }


def reset_web_session(
    *,
    session_manager: SessionManager,
    agent: Any,
    session_key: str,
    forget_data_agent_request_ids: list[str] | None = None,
) -> str:
    """Clear nanobot session for *session_key* (same logic as ``/new``)."""

    session = session_manager.get_or_create(session_key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    session_manager.save(session)
    session_manager.invalidate(session.key)
    if snapshot:
        agent._schedule_background(agent.memory_consolidator.archive_messages(snapshot))

    da = agent.tools.get("data_agent")
    if da is not None and hasattr(da, "forget_request_id"):
        seen: set[str] = set()
        for rid in forget_data_agent_request_ids or []:
            r = str(rid).strip()
            if r and r not in seen:
                seen.add(r)
                da.forget_request_id(r)

    return session_key


async def run_web_http_api_server(
    *,
    host: str,
    port: int,
    token: str,
    session_manager: SessionManager,
    agent: Any,
    web_channel_config: Any = None,
) -> None:
    """Run until the task is cancelled; then tear down the aiohttp runner."""

    async def handle_options(_request: web.Request) -> web.StreamResponse:
        resp = web.Response(status=204)
        for k, v in _cors_headers().items():
            resp.headers[k] = v
        return resp

    async def handle_reset(request: web.Request) -> web.StreamResponse:
        if token:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {token}":
                return web.json_response(
                    {"ok": False, "error": "unauthorized"},
                    status=401,
                    headers=_cors_headers(),
                )
        try:
            data = await request.json()
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        chat_id = str(data.get("chat_id") or "").strip()
        channel = str(data.get("channel") or "web").strip() or "web"
        auth_token = str(
            data.get("auth_token") or data.get("authToken") or ""
        ).strip()

        auth_base, me_path, auth_timeout = web_session_auth_settings(web_channel_config)

        session_key: str | None = None
        forget_ids: list[str] = []

        if auth_base and auth_token:
            uid = await resolve_web_session_user_id(
                base_url=auth_base,
                me_path=me_path,
                bearer_token=auth_token,
                timeout=auth_timeout,
            )
            if not uid:
                return web.json_response(
                    {
                        "ok": False,
                        "error": "auth_me_failed",
                        "detail": "Could not resolve user id from auth_token (check sessionAuthBaseUrl / sessionAuthMePath)",
                    },
                    status=502,
                    headers=_cors_headers(),
                )
            session_key = f"{channel}:{uid}"
            forget_ids.append(uid)
            if chat_id:
                forget_ids.append(chat_id)
        elif not chat_id:
            return web.json_response(
                {"ok": False, "error": "chat_id required"},
                status=400,
                headers=_cors_headers(),
            )
        else:
            if auth_base and not auth_token:
                logger.info(
                    "Web HTTP reset: sessionAuthBaseUrl set but no auth_token; "
                    "clearing legacy key {}:{} (prefer auth_token to clear user-scoped session)",
                    channel,
                    chat_id,
                )
            session_key = f"{channel}:{chat_id}"
            forget_ids.append(chat_id)

        try:
            key = reset_web_session(
                session_manager=session_manager,
                agent=agent,
                session_key=session_key,
                forget_data_agent_request_ids=forget_ids,
            )
        except Exception as e:
            logger.exception("Web HTTP API session reset failed")
            return web.json_response(
                {"ok": False, "error": f"{type(e).__name__}: {e}"},
                status=500,
                headers=_cors_headers(),
            )
        return web.json_response(
            {"ok": True, "key": key},
            headers=_cors_headers(),
        )

    app = web.Application()
    app.router.add_route("OPTIONS", "/api/session/reset", handle_options)
    app.router.add_post("/api/session/reset", handle_reset)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(
        "Web HTTP API: POST http://{}:{}/api/session/reset (Bearer token if web channel token set)",
        host,
        port,
    )
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        raise
    finally:
        await runner.cleanup()


def web_http_listen_config(config: Any) -> tuple[bool, str, int, str]:
    """Return (start_server, host, port, token). Server runs only if web + http API are enabled."""
    w = getattr(config.channels, "web", None)
    if w is None:
        return False, "0.0.0.0", 8766, ""
    if isinstance(w, dict):
        base = bool(w.get("enabled", False))
        http_on = bool(w.get("httpApiEnabled", False))
        return (
            base and http_on,
            str(w.get("host", "0.0.0.0")),
            int(w.get("httpApiPort", 8766)),
            str(w.get("token", "") or ""),
        )
    base = bool(getattr(w, "enabled", False))
    http_on = bool(getattr(w, "http_api_enabled", False))
    return (
        base and http_on,
        str(getattr(w, "host", "0.0.0.0")),
        int(getattr(w, "http_api_port", 8766)),
        str(getattr(w, "token", "") or ""),
    )
