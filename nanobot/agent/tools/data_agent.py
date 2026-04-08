"""DataAgent streaming tool — call an external DataAgent service via SSE."""

from __future__ import annotations

import json
import time
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage

# Per asyncio-task payload from inbound message ``metadata["data_agent"]`` (web / API clients).
_data_agent_request_overlay: ContextVar[dict[str, Any] | None] = ContextVar(
    "data_agent_request_overlay", default=None
)


def attach_data_agent_overlay_from_message(metadata: dict[str, Any] | None) -> None:
    """Attach QueryReq-shaped fields from channel message metadata for this request only.

    Web / HTTP clients may send ``metadata.data_agent`` (or ``dataAgent``) with keys such as
    ``request_id``, ``design_template_id``, ``assistant``, ``llm_param``, ``artifacts``, ``auth_token``, etc.
    Root-level ``metadata.auth_token`` / ``authToken`` is merged in as DataAgent Bearer (same as nested).
    These merge over ``config.json`` defaults when building the DataAgent request body.
    Cleared by :func:`detach_data_agent_overlay` after the agent turn.
    """
    if not metadata:
        _data_agent_request_overlay.set(None)
        return
    merged: dict[str, Any] = {}
    nested = metadata.get("data_agent") or metadata.get("dataAgent")
    if isinstance(nested, dict):
        merged.update(nested)
    root_tok = metadata.get("auth_token") or metadata.get("authToken")
    if root_tok and str(root_tok).strip() and "auth_token" not in merged:
        merged["auth_token"] = str(root_tok).strip()
    if merged:
        _data_agent_request_overlay.set(merged)
    else:
        _data_agent_request_overlay.set(None)


def detach_data_agent_overlay() -> None:
    _data_agent_request_overlay.set(None)


def _overlay() -> dict[str, Any] | None:
    return _data_agent_request_overlay.get()


def _merge_shallow_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if v is not None:
            out[k] = v
    return out


def _normalize_keys_camel_to_snake(d: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        nk = mapping.get(k, k)
        out[nk] = v
    return out


_LLM_PARAM_CAMEL = {
    "apiBase": "api_base",
    "apiKey": "api_key",
    "apiType": "api_type",
    "maxTokens": "max_tokens",
    "topP": "top_p",
    "frequencyPenalty": "frequency_penalty",
    "presencePenalty": "presence_penalty",
    "maxRetries": "max_retries",
    "responseFormat": "response_format",
    "toolChoice": "tool_choice",
    "maxContextTokens": "max_context_tokens",
}

_XMETRIC_CAMEL = {
    "xmetricApiBaseUrl": "xmetric_api_base_url",
    "xmetricApiEndpoint": "xmetric_api_endpoint",
    "xmetricParquetEndpoint": "xmetric_parquet_endpoint",
    "xmetricApiTimeout": "xmetric_api_timeout",
}

_ARTIFACT_CAMEL = {
    "objectName": "object_name",
}

_CONTEXT_ITEM_CAMEL = {
    "actionParams": "action_params",
}


def _normalize_llm_param(d: dict[str, Any]) -> dict[str, Any]:
    return _normalize_keys_camel_to_snake(d, _LLM_PARAM_CAMEL)


def _normalize_xmetric(d: dict[str, Any]) -> dict[str, Any]:
    return _normalize_keys_camel_to_snake(d, _XMETRIC_CAMEL)


def _normalize_artifact_item(a: Any) -> Any:
    if not isinstance(a, dict):
        return a
    return _normalize_keys_camel_to_snake(a, _ARTIFACT_CAMEL)


class DataAgentTool(Tool):
    """Query an external DataAgent service and stream the response to the user."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        base_url: str = "http://localhost:8002",
        chat_path: str = "/api/v1/agent/query/conversing",
        timeout: int = 120,
        assistant: str = "",
        llm_param: dict[str, Any] | None = None,
        xmetric_param: dict[str, Any] | None = None,
        auth_token: str = "",
        new_session_path: str = "/api/v1/agent/query/new_session",
    ):
        self._send = send_callback
        self._base_url = base_url.rstrip("/")
        self._chat_path = chat_path
        self._new_session_path = new_session_path
        self._timeout = timeout
        self._assistant = assistant
        self._llm_param = llm_param
        self._xmetric_param = xmetric_param
        self._auth_token = (auth_token or "").strip()
        self._channel = ""
        self._chat_id = ""
        self._user_message: str = ""
        # DataAgent server-side sessions keyed by request_id (we default to nanobot chat_id).
        self._sessions_started: set[str] = set()

    # ── context injection (same pattern as MessageTool / CronTool) ──

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    def set_user_message(self, message: str) -> None:
        """Store the user's original message so execute() can forward it faithfully."""
        self._user_message = (message or "").strip()

    def clear_user_message(self) -> None:
        self._user_message = ""

    def set_send_callback(self, cb: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        self._send = cb

    def forget_request_id(self, request_id: str) -> None:
        """Drop cached new_session state so the next call re-creates DataAgent server session."""
        self._sessions_started.discard(request_id)

    # ── tool schema ──

    @property
    def name(self) -> str:
        return "data_agent"

    @property
    def description(self) -> str:
        return (
            "调用 DataAgent 做数据分析、查数、报表、图表、或上传的 CSV/Excel 等文件分析。"
            "结果会通过流式输出直接发到用户界面（表格、图表、报告等）；你只会收到简短确认摘要。"
            "禁止因同一问题重复调用本工具。"
            "【instruction 极其重要】只写用户原话中与「数据」直接相关的那一段，"
            "逐字沿用用户的措辞与范围，禁止补充「包括但不限于…」、禁止臆造合同额/收入/成本/利润/同比环比/维度等你未被用户点名的指标。"
            "若用户一句话里既有查数又有其他任务（如定时任务），instruction 里只保留查数相关子句，其他任务用别的工具处理。"
            "正面示例：用户说「帮我看下去年10月经营情况」→ instruction 就写这句或等价最短摘录，不要展开成长串指标清单。"
            "反面示例：用户仅问经营情况，却写成「…包括但不限于合同额、收入、成本、与上月及去年同期对比」——禁止。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": (
                        "发往 DataAgent 的指令，必须短于或等于用户原话中能覆盖的范围："
                        "可从整句中摘录与数据分析相关的连续片段（去掉定时任务、发邮件等无关后半句），"
                        "但不要改写得更长、不要加用户没说的指标或对比维度。"
                        "例：用户「分析去年10月经营情况，并设成定时任务」→ 填「分析去年10月经营情况」。"
                        "例：用户「看下Q3销售额」→ 填「看下Q3销售额」，不要扩展成「Q3销售额、毛利、客单价…」。"
                    ),
                },
            },
            "required": ["instruction"],
        }

    # ── authentication ──

    def _auth_headers(self) -> dict[str, str]:
        """Bearer token for ``new_session`` and ``conversing`` (config and/or per-request overlay)."""
        headers: dict[str, str] = {}
        ov = _overlay() or {}
        token = (
            str(ov.get("auth_token") or ov.get("authToken") or "").strip()
            or self._auth_token
        )
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _ensure_data_agent_session(self, client: Any, request_id: str) -> str | None:
        """Call new_session once per request_id in this process before conversing.

        Returns an error message string on hard failure, or None on success.
        """
        if not self._new_session_path:
            return None
        if request_id in self._sessions_started:
            return None

        url = f"{self._base_url}{self._new_session_path}"
        headers = self._auth_headers()
        try:
            resp = await client.post(
                url,
                json={"request_id": request_id},
                headers=headers,
                timeout=30.0,
            )
        except Exception as e:
            logger.warning("DataAgent new_session request failed: {}: {}", type(e).__name__, e)
            return f"Error: DataAgent new_session failed — {type(e).__name__}: {e}"

        if resp.is_success:
            self._sessions_started.add(request_id)
            logger.debug("DataAgent new_session ok request_id={}", request_id)
            return None

        text = (resp.text or "")[:500]
        # Session may already exist (idempotent duplicate).
        if resp.status_code in (409, 400):
            low = text.lower()
            if any(
                s in low
                for s in ("already", "exist", "duplicate", "会话", "已存在")
            ):
                self._sessions_started.add(request_id)
                logger.debug("DataAgent new_session treated as existing request_id={}", request_id)
                return None

        logger.warning(
            "DataAgent new_session HTTP {} request_id={} body={}",
            resp.status_code,
            request_id,
            text,
        )
        return f"Error: DataAgent new_session HTTP {resp.status_code} — {text}"

    # ── execution ──

    async def execute(self, instruction: str, **kwargs: Any) -> str:
        try:
            import httpx
        except ImportError:
            return "Error: httpx package is required. Run: pip install httpx"

        # Guard against LLM fabricating metrics/dimensions the user never asked for.
        # If the LLM's instruction is SHORTER than (or equal to) the original message,
        # it likely extracted the data-relevant portion (e.g. from a multi-task message)
        # — trust it.  If LONGER, the LLM probably expanded/fabricated — use the original.
        if self._user_message:
            if len(instruction) <= len(self._user_message):
                effective_instruction = instruction
                if instruction != self._user_message:
                    logger.debug(
                        "DataAgent using LLM-extracted instruction (shorter): '{}' from '{}'",
                        instruction[:80],
                        self._user_message[:80],
                    )
            else:
                effective_instruction = self._user_message
                logger.debug(
                    "DataAgent ignoring LLM expansion, using original: '{}' (LLM wanted: '{}')",
                    self._user_message[:80],
                    instruction[:80],
                )
        else:
            effective_instruction = instruction

        ov = _overlay() or {}
        effective_rid = (
            str(ov.get("request_id") or ov.get("requestId") or "").strip()
            or (self._chat_id or "").strip()
        )
        if not effective_rid:
            return (
                "Error: DataAgent needs request_id. "
                "Send metadata.data_agent.request_id from the client, or use a channel with chat_id so it can default."
            )

        stream_id = f"da:{self._chat_id}:{time.time_ns()}"
        can_stream = bool(self._send and self._channel and self._chat_id)
        artifacts_seq_internal: list[dict[str, Any]] = []
        ask_help_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}

        body = self._build_request(effective_instruction, effective_rid)
        url = f"{self._base_url}{self._chat_path}"
        headers = self._auth_headers()
        logger.info(
            "DataAgent request: {} request_id={} instruction={}",
            url,
            effective_rid,
            effective_instruction[:80],
        )

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10, read=self._timeout, write=10, pool=10),
            ) as client:
                if err := await self._ensure_data_agent_session(client, effective_rid):
                    return err
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        parsed = self._parse_sse_line(line)
                        if parsed is None:
                            continue
                        delta, artifact, sse_obj = parsed
                        if delta:
                            if can_stream:
                                await self._push_delta(delta, stream_id, sse_obj)
                        if artifact:
                            art_type = str(artifact.get("__type") or "").strip()
                            if art_type in ("ask_help_json", "summary", "report_markdown"):
                                key = (
                                    str(artifact.get("title") or ""),
                                    str(artifact.get("entityid") or ""),
                                    str(artifact.get("entity_parent_id") or ""),
                                    int(artifact.get("entity_order") or 0),
                                )
                                if key in ask_help_by_key:
                                    ask_help_by_key[key]["content"] = (
                                        str(ask_help_by_key[key].get("content") or "")
                                        + str(artifact.get("content") or "")
                                    )
                                else:
                                    ask_help_by_key[key] = artifact
                                    artifacts_seq_internal.append(artifact)
                            else:
                                artifacts_seq_internal.append(artifact)

            if can_stream:
                await self._push_end(stream_id)

        except Exception as e:
            logger.warning("DataAgent request failed: {}: {}", type(e).__name__, e)
            if can_stream:
                await self._push_end(stream_id)
            return f"Error: DataAgent request failed — {type(e).__name__}: {e}"

        full_text = self._build_artifact_markdown(artifacts_seq_internal)

        if not full_text and not can_stream:
            return "DataAgent returned an empty response."
        llm_summary = self._build_llm_summary(artifacts_seq_internal, full_text, can_stream)
        return llm_summary

    # ── request building ──

    def _build_request(
        self,
        instruction: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a QueryReq-compatible JSON body (config + optional per-request metadata overlay)."""
        ov = _overlay() or {}

        assistant = (str(ov.get("assistant") or "").strip() or self._assistant)
        body: dict[str, Any] = {
            "instruction": instruction,
            "assistant": assistant,
        }
        if request_id:
            body["request_id"] = request_id

        dt = ov.get("design_template_id") or ov.get("designTemplateId")
        if dt:
            body["design_template_id"] = dt

        rtm = ov.get("report_template_markdown") or ov.get("reportTemplateMarkdown")
        if rtm:
            body["report_template_markdown"] = rtm

        lp: dict[str, Any] = {}
        if self._llm_param:
            lp = {k: v for k, v in self._llm_param.items() if v is not None}
        raw_lp = ov.get("llm_param") or ov.get("llmParam")
        if isinstance(raw_lp, dict) and raw_lp:
            lp = _merge_shallow_dicts(lp, _normalize_llm_param(raw_lp))
        if lp:
            body["llm_param"] = lp

        xp: dict[str, Any] = {}
        if self._xmetric_param:
            xp = {k: v for k, v in self._xmetric_param.items() if v is not None}
        raw_xp = ov.get("xmetric_param") or ov.get("xmetricParam")
        if isinstance(raw_xp, dict) and raw_xp:
            xp = _merge_shallow_dicts(xp, _normalize_xmetric(raw_xp))
        if xp:
            body["xmetric_param"] = xp

        arts = ov.get("artifacts")
        if isinstance(arts, list) and arts:
            body["artifacts"] = [_normalize_artifact_item(x) for x in arts]

        ctxs = ov.get("contexts")
        if isinstance(ctxs, list) and ctxs:
            body["contexts"] = [
                _normalize_keys_camel_to_snake(x, _CONTEXT_ITEM_CAMEL) if isinstance(x, dict) else x
                for x in ctxs
            ]

        return body

    _INLINE_CONTENT_TYPES = {"summary", "table_markdown", "report_markdown", "ask_help_json"}

    @staticmethod
    def _build_artifact_markdown(artifacts: list[dict[str, Any]]) -> str:
        """Build a Task → Step structured outline for LLM context.

        Content inclusion rules by ``__type``:
        * summary, table_markdown, report_markdown, ask_help_json → full content
        * report_json → metadata title only (already extracted upstream)
        * table, plotly → brief note (full data delivered to user via streaming)
        """
        if not artifacts:
            return ""

        nodes: dict[str, dict[str, Any]] = {}
        order_counter = 0
        root_ids: set[str] = set()

        def ensure_node(
            entity_id: str,
            entity_parent_id: str,
            entity_type: str,
            entity_order: int,
            entity_title: str,
        ) -> dict[str, Any]:
            nonlocal order_counter
            if entity_id not in nodes:
                nodes[entity_id] = {
                    "id": entity_id,
                    "parent_id": entity_parent_id,
                    "type": entity_type,
                    "order": entity_order,
                    "title": entity_title,
                    "items": [],
                    "children": [],
                    "_insert": order_counter,
                }
                order_counter += 1
            else:
                n = nodes[entity_id]
                if entity_parent_id:
                    n["parent_id"] = entity_parent_id
                if entity_type:
                    n["type"] = entity_type
                if entity_title:
                    n["title"] = entity_title
                n["order"] = entity_order
            return nodes[entity_id]

        for a in artifacts:
            entity_id = str(a.get("entityid") or "").strip()
            if not entity_id:
                continue
            entity_parent_id = str(a.get("entity_parent_id") or "").strip()
            entity_type = str(a.get("entity_type") or "").strip()
            entity_title = str(a.get("entity_title") or "").strip()
            try:
                entity_order = int(a.get("entity_order") or 0)
            except Exception:
                entity_order = 0

            node = ensure_node(
                entity_id=entity_id,
                entity_parent_id=entity_parent_id,
                entity_type=entity_type,
                entity_order=entity_order,
                entity_title=entity_title,
            )
            node["items"].append({
                "art_type": str(a.get("__type") or "").strip(),
                "title": str(a.get("title") or "").strip(),
                "content": str(a.get("content") or "").strip(),
                "_insert": order_counter,
            })
            order_counter += 1
            if not entity_parent_id:
                root_ids.add(entity_id)

        for node_id, node in list(nodes.items()):
            pid = str(node.get("parent_id") or "").strip()
            if pid and pid in nodes:
                nodes[pid]["children"].append(node_id)
            else:
                root_ids.add(node_id)

        def sort_key_node(node_id: str) -> tuple[int, int]:
            n = nodes[node_id]
            return (int(n.get("order") or 0), int(n.get("_insert") or 0))

        def sort_key_item(item: dict[str, Any]) -> int:
            return int(item.get("_insert") or 0)

        lines: list[str] = []

        def render_item(item: dict[str, Any], indent: str) -> None:
            art_type = item.get("art_type", "")
            title = item.get("title", "")
            content = item.get("content", "")

            if art_type in DataAgentTool._INLINE_CONTENT_TYPES:
                if content:
                    lines.append(f"{indent}{content}")
            elif art_type == "report_json":
                if content:
                    lines.append(f"{indent}[report] {content}")
            elif art_type == "table":
                lines.append(f"{indent}[table: {title or content or 'data'}] (delivered to user)")
            elif art_type == "plotly":
                lines.append(f"{indent}[chart: {title or content or 'chart'}] (delivered to user)")
            elif content:
                lines.append(f"{indent}{content}")

        def render(node_id: str, level: int) -> None:
            n = nodes[node_id]
            node_type = n.get("type", "")
            title = str(n.get("title") or "").strip()
            indent = "  " * level

            if title:
                prefix = "Task" if node_type == "task" else "Step" if node_type == "step" else ""
                lines.append(f"{indent}{prefix}: {title}" if prefix else f"{indent}{title}")

            item_indent = indent + "  "
            for item in sorted(n.get("items", []), key=sort_key_item):
                render_item(item, item_indent)

            for child_id in sorted(n.get("children", []), key=sort_key_node):
                render(child_id, level + 1)

        for rid in sorted(root_ids, key=sort_key_node):
            render(rid, 0)

        return "\n".join(lines).strip()

    # ── LLM result summary ──

    @staticmethod
    def _build_llm_summary(
        artifacts: list[dict[str, Any]],
        outline_text: str,
        streamed_to_user: bool,
    ) -> str:
        """Build a concise summary for the LLM context that confirms delivery.

        The full data (large tables, charts, report JSON) has already been pushed
        to the web client.  The LLM only needs to know *what* was delivered so it
        can answer follow-up questions without re-calling this tool.
        """
        type_counts: dict[str, int] = {}
        for a in artifacts:
            t = str(a.get("__type") or "unknown").strip()
            type_counts[t] = type_counts.get(t, 0) + 1

        type_labels = {
            "table": "data table(s)",
            "table_markdown": "table data preview(s)",
            "plotly": "chart(s)",
            "summary": "text summary(ies)",
            "report_json": "report(s)",
            "report_markdown": "markdown report(s)",
            "ask_help_json": "clarification request(s)",
        }

        delivered: list[str] = []
        for t, count in type_counts.items():
            label = type_labels.get(t, t)
            delivered.append(f"{count} {label}")

        parts: list[str] = []
        if streamed_to_user:
            parts.append(
                "[DELIVERED] The following results have been streamed directly to the user's screen. "
                "The user can already see all tables, charts, and reports. "
                "Do NOT call data_agent again for this request."
            )
        if delivered:
            parts.append("Artifacts delivered: " + ", ".join(delivered) + ".")
        if outline_text:
            parts.append("Content outline:\n" + outline_text)

        if not parts:
            return "DataAgent returned an empty response."

        return "\n\n".join(parts)

    @staticmethod
    def _parse_sse_line(
        line: str,
    ) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None] | None:
        """Parse one SSE line from FastAPI EventSourceResponse.

        Returns (delta_text, artifact_dict, raw_json_obj) or None if the line should be skipped.
        ``raw_json_obj`` is the parsed object when it is a dict; otherwise None.
        """
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return None
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, AttributeError):
            return (payload, None, None) if payload else None
        
        def _as_str(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, str):
                return v
            if isinstance(v, (dict, list)):
                return json.dumps(v, ensure_ascii=False)
            return str(v)

        allowed_types = {
            "table",
            "table_markdown",
            "plotly",
            "summary",
            "report_json",
            "report_markdown",
            "ask_help_json",
        }

        if not isinstance(obj, dict):
            return None

        payload_obj = obj.get("payload") or {}
        msg_type = ""
        payload_content: Any = None
        if isinstance(payload_obj, dict):
            msg_type = str(payload_obj.get("msg_type") or "").strip()
            payload_content = payload_obj.get("content")

        # Keep delta only for plain text chunks.
        delta: str | None = payload_content if isinstance(payload_content, str) else None

        artifact_dict: dict[str, Any] | None = None
        if msg_type in allowed_types:
            entity_obj = obj.get("entity") if isinstance(obj.get("entity"), dict) else {}
            title = str(payload_content.get("title") or "").strip() if isinstance(payload_content, dict) else ""
            entityid = str(entity_obj.get("id") or "").strip()
            entity_parent_id = str(entity_obj.get("parent_id") or "").strip()
            entity_type = str(entity_obj.get("type") or "").strip()
            entity_title = str(entity_obj.get("title") or "").strip()
            try:
                entity_order = int(entity_obj.get("order") or 0)
            except Exception:
                entity_order = 0

            content = ""
            if msg_type == "plotly":
                if isinstance(payload_content, dict):
                    att = payload_content.get("attachment") if isinstance(payload_content.get("attachment"), dict) else {}
                    content = str((att or {}).get("desc") or payload_content.get("description") or "")

            elif msg_type == "table":
                if isinstance(payload_content, dict):
                    att = payload_content.get("attachment") if isinstance(payload_content.get("attachment"), dict) else {}
                    content = str((att or {}).get("desc") or payload_content.get("description") or "")

            elif msg_type in ("summary", "table_markdown", "report_markdown", "ask_help_json"):
                content = _as_str(payload_content)
            elif msg_type == "report_json":
                if isinstance(payload_content, dict):
                    md = payload_content.get("metadata") if isinstance(payload_content.get("metadata"), dict) else {}
                    content = _as_str((md or {}).get("title") or "")
                else:
                    content = _as_str(payload_content)

            artifact_dict = {
                "__type": msg_type,
                "title": title,
                "entityid": entityid,
                "entity_type": entity_type,
                "entity_title": entity_title,
                "entity_parent_id": entity_parent_id,
                "entity_order": entity_order,
                "content": content,
            }

        return (delta, artifact_dict, obj)

    # ── stream helpers ──

    async def _push_delta(
        self,
        delta: str,
        stream_id: str,
        sse_obj: dict[str, Any] | None = None,
    ) -> None:
        if self._send:
            meta: dict[str, Any] = {**(sse_obj or {}), "_stream_delta": True, "_stream_id": stream_id}
            await self._send(OutboundMessage(
                channel=self._channel,
                chat_id=self._chat_id,
                content=delta,
                metadata=meta,
            ))

    async def _push_end(self, stream_id: str) -> None:
        if self._send:
            await self._send(OutboundMessage(
                channel=self._channel,
                chat_id=self._chat_id,
                content="",
                metadata={"_stream_end": True, "_stream_id": stream_id},
            ))
