"""WebSocket channel for browser/web clients with optional streaming support."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base

try:
    import websockets
except Exception:  # pragma: no cover - import error handled in start()
    websockets = None


class WebConfig(Base):
    """Web channel configuration."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8765
    path: str = "/ws"
    token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    streaming: bool = True


@dataclass(eq=False)
class _ClientSession:
    """Connection metadata for one browser/websocket client."""

    ws: Any
    sender_id: str | None = None
    chat_id: str | None = None
    authed: bool = False


class WebChannel(BaseChannel):
    """A bidirectional WebSocket channel for web frontends."""

    name = "web"
    display_name = "Web"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WebConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = WebConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: WebConfig = config
        self._clients: set[_ClientSession] = set()
        self._clients_by_chat: dict[str, set[_ClientSession]] = {}
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start WebSocket server and keep serving until stopped."""
        if websockets is None:
            logger.error("Web channel requires websockets package")
            return

        self._running = True
        self._stop_event.clear()

        async def _handler(ws):
            await self._handle_connection(ws)

        logger.info("Starting Web channel on ws://{}:{}{}", self.config.host, self.config.port, self.config.path)
        async with websockets.serve(_handler, self.config.host, self.config.port):
            await self._stop_event.wait()

    async def stop(self) -> None:
        """Stop server and close active connections."""
        self._running = False
        self._stop_event.set()
        async with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                await client.ws.close(code=1001, reason="Server stopping")
            except Exception:
                pass

    async def send(self, msg: OutboundMessage) -> None:
        """Send a complete reply to web clients."""
        payload = {
            "type": "message",
            "chat_id": msg.chat_id,
            "content": msg.content,
            "media": msg.media,
            "metadata": msg.metadata,
        }
        await self._broadcast(msg.chat_id, payload)

    async def send_delta(self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None) -> None:
        """Send streaming delta/end events to web clients."""
        meta = metadata or {}
        payload = {
            "type": "stream_end" if meta.get("_stream_end") else "stream_delta",
            "chat_id": chat_id,
            "delta": delta,
            "metadata": meta,
        }
        await self._broadcast(chat_id, payload)

    async def _handle_connection(self, ws: Any) -> None:
        """Handle one websocket client lifecycle."""
        req_path = getattr(ws, "path", None)
        if req_path is None:
            req = getattr(ws, "request", None)
            req_path = getattr(req, "path", None)
        if req_path and req_path != self.config.path:
            await ws.close(code=4404, reason="Not Found")
            return

        client = _ClientSession(ws=ws, authed=not bool(self.config.token))
        await self._register_client(client)

        try:
            await self._safe_send(ws, {
                "type": "hello",
                "channel": self.name,
                "requires_auth": bool(self.config.token),
                "protocol": {
                    "inbound": ["auth", "message"],
                    "outbound": ["message", "stream_delta", "stream_end", "error"],
                },
            })

            async for raw in ws:
                await self._handle_client_payload(client, raw)
        except Exception as e:
            logger.debug("Web client disconnected: {}", e)
        finally:
            await self._unregister_client(client)

    async def _handle_client_payload(self, client: _ClientSession, raw: Any) -> None:
        """Parse and process one client payload."""
        try:
            payload = json.loads(raw) if isinstance(raw, str) else {}
        except Exception:
            await self._safe_send(client.ws, {"type": "error", "error": "invalid_json"})
            return

        msg_type = str(payload.get("type") or "")
        if msg_type == "auth":
            await self._handle_auth(client, payload)
            return

        if not client.authed:
            await self._safe_send(client.ws, {"type": "error", "error": "unauthorized"})
            return

        if msg_type != "message":
            await self._safe_send(client.ws, {"type": "error", "error": "unknown_type"})
            return

        sender_id = str(payload.get("sender_id") or client.sender_id or "web-user")
        chat_id = str(payload.get("chat_id") or client.chat_id or sender_id)
        content = str(payload.get("content") or "").strip()
        media = payload.get("media") if isinstance(payload.get("media"), list) else []
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

        if not content and not media:
            await self._safe_send(client.ws, {"type": "error", "error": "empty_message"})
            return

        client.sender_id = sender_id
        if client.chat_id != chat_id:
            await self._bind_chat(client, chat_id)

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=media,
            metadata=metadata,
        )

    async def _handle_auth(self, client: _ClientSession, payload: dict[str, Any]) -> None:
        token = str(payload.get("token") or "")
        if self.config.token and token != self.config.token:
            await self._safe_send(client.ws, {"type": "error", "error": "bad_token"})
            await client.ws.close(code=4401, reason="Unauthorized")
            return
        client.authed = True
        if chat_id := payload.get("chat_id"):
            await self._bind_chat(client, str(chat_id))
        if sender_id := payload.get("sender_id"):
            client.sender_id = str(sender_id)
        await self._safe_send(client.ws, {"type": "auth_ok"})

    async def _register_client(self, client: _ClientSession) -> None:
        async with self._lock:
            self._clients.add(client)

    async def _unregister_client(self, client: _ClientSession) -> None:
        async with self._lock:
            self._clients.discard(client)
            if client.chat_id and client.chat_id in self._clients_by_chat:
                bucket = self._clients_by_chat[client.chat_id]
                bucket.discard(client)
                if not bucket:
                    self._clients_by_chat.pop(client.chat_id, None)

    async def _bind_chat(self, client: _ClientSession, chat_id: str) -> None:
        async with self._lock:
            if client.chat_id and client.chat_id in self._clients_by_chat:
                old = self._clients_by_chat[client.chat_id]
                old.discard(client)
                if not old:
                    self._clients_by_chat.pop(client.chat_id, None)
            client.chat_id = chat_id
            self._clients_by_chat.setdefault(chat_id, set()).add(client)

    async def _broadcast(self, chat_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients_by_chat.get(chat_id, set()) or self._clients)
        if not targets:
            logger.debug("Web channel: no clients for chat_id={} payload={}", chat_id, payload.get("type"))
            return

        dead: list[_ClientSession] = []
        for client in targets:
            try:
                await self._safe_send(client.ws, payload)
            except Exception:
                dead.append(client)
        for client in dead:
            await self._unregister_client(client)

    @staticmethod
    async def _safe_send(ws: Any, payload: dict[str, Any]) -> None:
        await ws.send(json.dumps(payload, ensure_ascii=False))
