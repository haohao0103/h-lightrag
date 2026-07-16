"""HugeGraph REST async client + shared helpers for LightRAG.

Uses the HugeGraph REST API (NOT the Gremlin endpoint). HugeGraph 1.7.x
disables the gremlin-groovy script engine by default for security, so the
REST vertices/edges/schema endpoints are the only stable programmatic path.
This mirrors what the ``pyhugegraph`` SDK does internally for CRUD.

Key endpoints (all under ``{base_url}``):

  Schema:
    GET  /graphs/{g}/schema
    POST /graphs/{g}/schema        body {"schema": "<groovy DSL>"}
  Vertices:
    GET    /graphs/{g}/graph/vertices?label=X&limit=N
    GET    /graphs/{g}/graph/vertices/{id}
    POST   /graphs/{g}/graph/vertices       body {"label","properties","id"}
    PUT    /graphs/{g}/graph/vertices/{id}  body {"label","properties"}
    DELETE /graphs/{g}/graph/vertices/{id}
  Edges:
    GET    /graphs/{g}/graph/edges?vertex_id="X"&direction=BOTH&limit=N
    GET    /graphs/{g}/graph/edges/{id}
    POST   /graphs/{g}/graph/edges          body {"label","source","target","properties"}
    DELETE /graphs/{g}/graph/edges/{id}

Vertex id strategy for our labels: ``CUSTOMIZE_STRING`` so the LightRAG
``node_id`` (entity name, may contain CJK / spaces) is used verbatim as the
HugeGraph vertex id. The id is URL-encoded when placed in the path.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import quote

import httpx

from lightrag.utils import logger

# Schema label names shared between client and adapters
ENTITY_LABEL = "lightrag_entity"
RELATION_LABEL = "lightrag_relation"
DOC_STATUS_LABEL = "lightrag_doc_status"

# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def sanitize_label(name: str) -> str:
    """Make a string safe to use as a HugeGraph label name."""
    cleaned = _LABEL_SAFE.sub("_", name).strip("_")
    if not cleaned:
        cleaned = "default"
    if not re.match(r"[A-Za-z_]", cleaned[0]):
        cleaned = f"ns_{cleaned}"
    return cleaned


def cap_str(s: Any, limit: int = 32000) -> str:
    """Cap a string to ``limit`` chars (hugegraph TEXT property has no hard
    cap, but we keep a sane upper bound to avoid bloating requests)."""
    if s is None:
        return ""
    text = s if isinstance(s, str) else str(s)
    return text[:limit] if len(text) > limit else text


def flatten_vertex(v: dict) -> dict:
    """Normalize a HugeGraph vertex REST object to a flat dict.

    Input shape: {"id": "...", "label": "...", "type": "vertex",
                  "properties": {"k": v, ...}}
    Output: {"id": "...", "label": "...", <k>: v, ...}
    """
    flat: dict[str, Any] = {
        "id": v.get("id"),
        "label": v.get("label"),
    }
    props = v.get("properties") or {}
    for k, val in props.items():
        flat[k] = val
    return flat


def flatten_edge(e: dict) -> dict:
    """Normalize a HugeGraph edge REST object to a flat dict.

    HG 1.7.x edge response uses ``outV`` / ``inV`` for endpoints (not
    ``source`` / ``target``). We surface both names for downstream compat.
    """
    out_v = e.get("outV")
    in_v = e.get("inV")
    flat: dict[str, Any] = {
        "id": e.get("id"),
        "label": e.get("label"),
        "source": out_v,  # alias for LightRAG compatibility
        "target": in_v,   # alias for LightRAG compatibility
        "outV": out_v,
        "inV": in_v,
    }
    props = e.get("properties") or {}
    for k, val in props.items():
        flat[k] = val
    return flat


# ---------------------------------------------------------------------------
# REST client
# ---------------------------------------------------------------------------


class HugeGraphClient:
    """Async wrapper around HugeGraph REST endpoints."""

    def __init__(
        self,
        base_url: str,
        graph: str,
        user: str,
        pwd: str,
        graphspace: str | None,
        workspace: str,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.graph = graph
        self.user = user
        self.pwd = pwd
        self.graphspace = (graphspace or "").strip() or None
        self.workspace = workspace or "base"
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                auth=(self.user, self.pwd) if self.user else None,
                headers={"Accept-Encoding": "identity", "Accept": "application/json"},
            )
        await self._ensure_schema()

    async def finalize(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- url helpers ----------------------------------------------------------

    def _graph_prefix(self) -> str:
        if self.graphspace:
            return f"/graphspaces/{self.graphspace}/graphs/{self.graph}"
        return f"/graphs/{self.graph}"

    @staticmethod
    def _enc_id(vid: str) -> str:
        """URL-encode a vertex id for use in a path segment.

        HugeGraph 1.7.x requires customize-string ids in the REST path to be
        JSON string literals -- i.e. wrapped in double quotes. A bare ``Apple``
        is rejected with "must be formatted as Number/String/UUID". So we wrap
        the raw id in double quotes and then percent-encode the whole thing.
        """
        import json as _json
        return quote(_json.dumps(str(vid), ensure_ascii=False), safe="")

    # -- schema ---------------------------------------------------------------

    async def _schema_exists(self, kind: str, name: str) -> bool:
        """Check if a propertykey/vertexlabel/edgelabel/indexlabel exists."""
        url = f"{self._graph_prefix()}/schema/{kind}/{name}"
        resp = await self._client.get(url)
        return resp.status_code == 200

    async def _create_property_key(self, name: str, data_type: str) -> None:
        if await self._schema_exists("propertykeys", name):
            return
        url = f"{self._graph_prefix()}/schema/propertykeys"
        body = {"name": name, "data_type": data_type, "cardinality": "SINGLE"}
        resp = await self._client.post(url, json=body)
        if resp.status_code not in (200, 201, 202):
            logger.warning(f"propertykey {name} create: {resp.status_code} {resp.text[:200]}")

    async def _create_vertex_label(
        self, name: str, properties: list[str], id_strategy: str = "CUSTOMIZE_STRING"
    ) -> None:
        if await self._schema_exists("vertexlabels", name):
            return
        url = f"{self._graph_prefix()}/schema/vertexlabels"
        body = {
            "name": name,
            "properties": properties,
            "id_strategy": id_strategy,
            "primary_keys": [],
            # All properties are nullable so upserts can set only a subset.
            "nullable_keys": properties,
        }
        resp = await self._client.post(url, json=body)
        if resp.status_code not in (200, 201, 202):
            logger.warning(f"vertexlabel {name} create: {resp.status_code} {resp.text[:200]}")

    async def _create_edge_label(
        self,
        name: str,
        source_label: str,
        target_label: str,
        properties: list[str],
    ) -> None:
        if await self._schema_exists("edgelabels", name):
            return
        url = f"{self._graph_prefix()}/schema/edgelabels"
        body = {
            "name": name,
            "source_label": source_label,
            "target_label": target_label,
            "properties": properties,
            # All properties nullable so upserts can set only a subset.
            "nullable_keys": properties,
            "frequency": "SINGLE",
        }
        resp = await self._client.post(url, json=body)
        if resp.status_code not in (200, 201, 202):
            logger.warning(f"edgelabel {name} create: {resp.status_code} {resp.text[:200]}")

    async def _create_index_label(
        self, name: str, base_label: str, field: str, index_type: str = "SECONDARY"
    ) -> None:
        if await self._schema_exists("indexlabels", name):
            return
        url = f"{self._graph_prefix()}/schema/indexlabels"
        body = {
            "name": name,
            "base_type": "VERTEX_LABEL",
            "base_value": base_label,
            "index_type": index_type,
            "fields": [field],
        }
        resp = await self._client.post(url, json=body)
        if resp.status_code not in (200, 201, 202):
            logger.warning(f"indexlabel {name} create: {resp.status_code} {resp.text[:200]}")

    async def _ensure_schema(self) -> None:
        """Create all propertykeys / vertexlabels / edgelabels / indexlabels.

        Each creation is guarded by an existence check so re-running is safe.
        Uses the per-resource JSON endpoints (POST /schema/propertykeys etc.)
        which is the only schema path supported by HugeGraph 1.7.x REST.
        """
        # property keys
        pk_specs = [
            ("entity_type", "TEXT"),
            ("description", "TEXT"),
            ("source_id", "TEXT"),
            ("content", "TEXT"),
            ("kv_value", "TEXT"),
            ("relation_type", "TEXT"),
            ("weight", "DOUBLE"),
            ("doc_status", "TEXT"),
            ("content_hash", "TEXT"),
            ("file_path", "TEXT"),
            ("file_basename", "TEXT"),
            ("track_id", "TEXT"),
            ("created_at", "LONG"),
            ("updated_at", "LONG"),
            ("chunk_count", "INT"),
            ("content_length", "LONG"),
        ]
        for name, dt in pk_specs:
            await self._create_property_key(name, dt)

        # vertex labels
        await self._create_vertex_label(
            "lightrag_entity",
            ["entity_type", "description", "source_id", "content"],
        )
        await self._create_vertex_label(
            "lightrag_doc_status",
            [
                "doc_status", "content_hash", "file_path", "file_basename",
                "track_id", "created_at", "updated_at", "chunk_count",
                "content_length", "kv_value",
            ],
        )

        # edge labels
        await self._create_edge_label(
            "lightrag_relation",
            "lightrag_entity",
            "lightrag_entity",
            ["relation_type", "description", "source_id", "weight", "content"],
        )

        # index labels
        await self._create_index_label("entity_by_entity_type", "lightrag_entity", "entity_type")
        await self._create_index_label("doc_status_by_status", "lightrag_doc_status", "doc_status")
        await self._create_index_label("doc_status_by_content_hash", "lightrag_doc_status", "content_hash")
        await self._create_index_label("doc_status_by_file_path", "lightrag_doc_status", "file_path")
        await self._create_index_label("doc_status_by_file_basename", "lightrag_doc_status", "file_basename")

    async def ensure_kv_label(self, namespace: str) -> str:
        """Lazily create the vertex label for a KV namespace; return label name."""
        label = f"lightrag_kv_{sanitize_label(namespace)}"
        await self._create_property_key("kv_value", "TEXT")
        await self._create_vertex_label(label, ["kv_value"])
        await self._create_index_label(f"{label}_by_kv_value", label, "kv_value")
        return label

    # -- vertex REST ----------------------------------------------------------

    async def get_vertex(self, vid: str) -> dict | None:
        """Get a vertex by id (any label). Returns the raw REST object or None."""
        url = f"{self._graph_prefix()}/graph/vertices/{self._enc_id(vid)}"
        resp = await self._client.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def get_vertex_by_label(self, vid: str, label: str) -> dict | None:
        v = await self.get_vertex(vid)
        if v is None:
            return None
        if v.get("label") != label:
            return None
        return v

    async def upsert_vertex(
        self, label: str, vid: str, properties: dict[str, Any]
    ) -> dict:
        """Upsert a vertex with a customize-string id.

        Uses POST (create-or-update semantics in HugeGraph 1.x: if the id
        already exists, the vertex is updated with the new properties).
        """
        url = f"{self._graph_prefix()}/graph/vertices"
        body = {
            "label": label,
            "id": str(vid),
            "properties": {k: v for k, v in properties.items() if v is not None},
        }
        resp = await self._client.post(
            url, json=body, headers={"Content-Type": "application/json"}
        )
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(
                f"upsert_vertex({label},{vid}) HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            return resp.json()
        except Exception:  # noqa: BLE001
            return {}

    async def delete_vertex(self, vid: str, label: str | None = None) -> bool:
        """Delete a vertex by id. If label is provided, only delete vertices
        matching that label (HG 1.7.0 supports ?label= query param)."""
        enc = self._enc_id(vid)
        url = f"{self._graph_prefix()}/graph/vertices/{enc}"
        if label:
            url += f"?label={label}"
        resp = await self._client.delete(url)
        return resp.status_code in (200, 204)

    async def list_vertices(
        self,
        label: str | None = None,
        limit: int = 1000,
        properties: dict | None = None,
    ) -> list[dict]:
        """List vertices, optionally filtered by label and secondary-indexed props.

        The ``properties`` query param must be a JSON object string (e.g.
        ``{"doc_status":"processed"}``), NOT ``key=value`` pairs. This matches
        the pyhugegraph SDK convention.
        """
        params: list[tuple[str, str]] = [("limit", str(limit))]
        if label:
            params.append(("label", label))
        if properties:
            params.append(("properties", json.dumps(properties, ensure_ascii=False)))
        url = f"{self._graph_prefix()}/graph/vertices"
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("vertices", []) or []

    async def count_vertices(self, label: str | None = None) -> int:
        # HugeGraph REST does not expose a direct count; we fetch limit=1 and
        # read the summary if present, otherwise fall back to listing.
        params = [("limit", "1")]
        if label:
            params.append(("label", label))
        url = f"{self._graph_prefix()}/graph/vertices"
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        # If the response carries a total, use it; else use len(vertices).
        if "total" in data:
            return int(data["total"])
        return len(data.get("vertices", []) or [])

    # -- edge REST ------------------------------------------------------------

    async def get_edges_between(
        self, src_id: str, tgt_id: str, label: str | None = None
    ) -> list[dict]:
        """List edges between two vertices (both directions).

        Uses ``format_vertex_id`` semantics: the ``vertex_id`` query param
        must be a JSON string literal (e.g. ``"Apple"``), matching what
        ``GET /vertices/{id}`` expects in the path.
        """
        params: list[tuple[str, str]] = [
            ("vertex_id", json.dumps(str(src_id), ensure_ascii=False)),
            ("direction", "BOTH"),
            ("limit", "1000"),
        ]
        if label:
            params.append(("label", label))
        url = f"{self._graph_prefix()}/graph/edges"
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        edges = resp.json().get("edges", []) or []
        # filter to those whose other endpoint == tgt_id
        # HG 1.7.x edge response uses outV/inV (not source/target)
        result: list[dict] = []
        for e in edges:
            out_v = str(e.get("outV", ""))
            in_v = str(e.get("inV", ""))
            if out_v == str(src_id) and in_v == str(tgt_id):
                result.append(e)
            elif out_v == str(tgt_id) and in_v == str(src_id):
                result.append(e)
        return result

    async def list_edges_of(
        self,
        src_id: str,
        direction: str = "BOTH",
        label: str | None = None,
        limit: int = 10000,
    ) -> list[dict]:
        """List edges incident to a vertex."""
        params: list[tuple[str, str]] = [
            ("vertex_id", json.dumps(str(src_id), ensure_ascii=False)),
            ("direction", direction),
            ("limit", str(limit)),
        ]
        if label:
            params.append(("label", label))
        url = f"{self._graph_prefix()}/graph/edges"
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json().get("edges", []) or []

    async def upsert_edge(
        self,
        label: str,
        src_id: str,
        tgt_id: str,
        properties: dict[str, Any],
    ) -> dict:
        """Upsert an edge. HugeGraph identifies edges by (label, source, target);
        re-POSTing the same edge updates its properties.

        1.7.x REST uses ``outV`` / ``inV`` / ``outVLabel`` / ``inVLabel`` field
        names (NOT ``source`` / ``target``).
        """
        url = f"{self._graph_prefix()}/graph/edges"
        body = {
            "label": label,
            # HG 1.7.x expects vertex ids in edge body to be the raw typed
            # value (string stays a JSON string, not bare). Sending the python
            # str lets httpx/json serialize it as a JSON string literal which
            # the server accepts.
            "outV": str(src_id),
            "inV": str(tgt_id),
            "outVLabel": ENTITY_LABEL if label == RELATION_LABEL else "",
            "inVLabel": ENTITY_LABEL if label == RELATION_LABEL else "",
            "properties": {k: v for k, v in properties.items() if v is not None},
        }
        resp = await self._client.post(
            url, json=body, headers={"Content-Type": "application/json"}
        )
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(
                f"upsert_edge({label},{src_id}->{tgt_id}) HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            return resp.json()
        except Exception:  # noqa: BLE001
            return {}

    async def delete_edge(self, edge_id: str) -> bool:
        url = f"{self._graph_prefix()}/graph/edges/{self._enc_id(edge_id)}"
        resp = await self._client.delete(url)
        return resp.status_code in (200, 204)

    async def delete_edges_between(
        self, src_id: str, tgt_id: str, label: str | None = None
    ) -> int:
        edges = await self.get_edges_between(src_id, tgt_id, label=label)
        deleted = 0
        for e in edges:
            eid = e.get("id")
            if eid is not None and await self.delete_edge(str(eid)):
                deleted += 1
        return deleted


# ---------------------------------------------------------------------------
# connection settings + client cache
# ---------------------------------------------------------------------------


def hg_settings_from_env() -> dict:
    return {
        "url": os.environ.get("HUGEGRAPH_URL", "http://127.0.0.1:8080"),
        "graph": os.environ.get("HUGEGRAPH_GRAPH", "hugegraph"),
        "user": os.environ.get("HUGEGRAPH_USER", ""),
        "pwd": os.environ.get("HUGEGRAPH_PWD", ""),
        "graphspace": os.environ.get("HUGEGRAPH_GRAPHSPACE", ""),
    }


_CLIENT_CACHE: dict[str, HugeGraphClient] = {}


async def get_client(workspace: str) -> HugeGraphClient:
    settings = hg_settings_from_env()
    cache_key = (
        f"{settings['url']}|{settings['graph']}|{settings['graphspace']}|{workspace}"
    )
    if cache_key not in _CLIENT_CACHE:
        client = HugeGraphClient(
            base_url=settings["url"],
            graph=settings["graph"],
            user=settings["user"],
            pwd=settings["pwd"],
            graphspace=settings["graphspace"],
            workspace=workspace,
        )
        await client.initialize()
        _CLIENT_CACHE[cache_key] = client
    return _CLIENT_CACHE[cache_key]
