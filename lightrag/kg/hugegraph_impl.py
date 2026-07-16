"""HugeGraph storage adapters for LightRAG (REST-based).

Three storage implementations backed by Apache HugeGraph server REST API:

* ``HugeGraphGraphStorage``     -- knowledge graph (entities + relations)
* ``HugeGraphKVStorage``        -- key/value namespaces (full_docs / chunks / ...)
* ``HugeGraphDocStatusStorage`` -- document processing status tracking

Vector storage is intentionally NOT implemented here: HugeGraph 1.7.x has no
first-class vector index. Configure LightRAG with ``NanoVectorDBStorage``
(the default) or any other vector backend when using the HugeGraph
graph/KV/doc-status backends.

Why REST not Gremlin: HugeGraph 1.7.x disables the gremlin-groovy script
engine by default (security hardening). The REST vertices/edges/schema
endpoints are the only stable programmatic path and match what the
``pyhugegraph`` SDK uses internally for CRUD.

Environment variables (mirror hugegraph-llm conventions):

    HUGEGRAPH_URL          default http://127.0.0.1:8080
    HUGEGRAPH_GRAPH        default hugegraph
    HUGEGRAPH_USER          default ""
    HUGEGRAPH_PWD           default ""
    HUGEGRAPH_GRAPHSPACE    default ""
    HUGEGRAPH_WORKSPACE     default base  (multi-tenant isolation key)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from lightrag.kg.hugegraph_client import (
    HugeGraphClient,
    cap_str,
    flatten_edge,
    flatten_vertex,
    get_client,
    sanitize_label,
)
from lightrag.utils import logger

# Separator used by LightRAG for edge keys in batch dicts
EDGE_KEY_SEP = "\t"

ENTITY_LABEL = "lightrag_entity"
RELATION_LABEL = "lightrag_relation"
DOC_STATUS_LABEL = "lightrag_doc_status"


def _to_timestamp(value: Any) -> int:
    """Convert a value to unix timestamp (int).

    Handles: int (passthrough), float (truncate), ISO timestamp string
    (e.g. "2026-07-15T09:57:28+00:00"), None (→ 0).
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        # Try int first
        try:
            return int(value)
        except ValueError:
            pass
        # Try ISO timestamp
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except (ValueError, TypeError):
            return 0
    return 0
DOC_STATUS_LABEL = "lightrag_doc_status"


# ---------------------------------------------------------------------------
# Graph storage
# ---------------------------------------------------------------------------


@dataclass
class HugeGraphGraphStorage(BaseGraphStorage):
    """Knowledge-graph storage backed by HugeGraph REST.

    Vertex id strategy: ``CUSTOMIZE_STRING`` -- the LightRAG ``node_id``
    (entity name) is used verbatim as the HugeGraph vertex id.
    """

    def __init__(
        self,
        namespace: str,
        global_config: dict,
        embedding_func,
        workspace: str | None = None,
    ) -> None:
        ws_env = os.environ.get("HUGEGRAPH_WORKSPACE")
        if ws_env and ws_env.strip():
            workspace = ws_env
        if not workspace or not str(workspace).strip():
            workspace = "base"
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver: HugeGraphClient | None = None

    async def initialize(self) -> None:
        self._driver = await get_client(self.workspace)

    async def finalize(self) -> None:
        self._driver = None

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        try:
            assert self._driver is not None
            # drop all edges then vertices by label
            edges = await self._driver.list_edges_of(
                "", direction="BOTH", label=RELATION_LABEL, limit=100000
            ) if False else []
            # HugeGraph REST delete-by-label is not supported; we list+delete.
            # For safety we delete vertices which cascades to their edges.
            verts = await self._driver.list_vertices(label=ENTITY_LABEL, limit=100000)
            for v in verts:
                vid = v.get("id")
                if vid is not None:
                    await self._driver.delete_vertex(str(vid))
            return {"status": "success", "message": "data dropped"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": str(e)}

    # -- existence ------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        assert self._driver is not None
        v = await self._driver.get_vertex_by_label(node_id, ENTITY_LABEL)
        return v is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        assert self._driver is not None
        edges = await self._driver.get_edges_between(
            source_node_id, target_node_id, label=RELATION_LABEL
        )
        return bool(edges)

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        result: set[str] = set()
        for nid in node_ids:
            if await self.has_node(nid):
                result.add(nid)
        return result

    # -- degree ---------------------------------------------------------------

    async def node_degree(self, node_id: str) -> int:
        assert self._driver is not None
        edges = await self._driver.list_edges_of(
            node_id, direction="BOTH", label=RELATION_LABEL
        )
        # dedup by edge id (BOTH may return each edge once already)
        seen = {e.get("id") for e in edges}
        return len(seen)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return int(await self.node_degree(src_id)) + int(await self.node_degree(tgt_id))

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        return {nid: await self.node_degree(nid) for nid in node_ids}

    async def edge_degrees_batch(
        self, edges: list[tuple[str, str]]
    ) -> dict[str, int]:
        return {
            f"{src}{EDGE_KEY_SEP}{tgt}": await self.edge_degree(src, tgt)
            for src, tgt in edges
        }

    # -- read -----------------------------------------------------------------

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        assert self._driver is not None
        v = await self._driver.get_vertex_by_label(node_id, ENTITY_LABEL)
        if v is None:
            return None
        return flatten_vertex(v)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, Any] | None:
        assert self._driver is not None
        edges = await self._driver.get_edges_between(
            source_node_id, target_node_id, label=RELATION_LABEL
        )
        if not edges:
            return None
        return flatten_edge(edges[0])

    async def get_node_edges(
        self, source_node_id: str
    ) -> list[tuple[str, str]] | None:
        assert self._driver is not None
        edges = await self._driver.list_edges_of(
            source_node_id, direction="BOTH", label=RELATION_LABEL
        )
        # HG 1.7.x edge response uses outV/inV (not source/target)
        result: list[tuple[str, str]] = []
        for e in edges:
            out_v = str(e.get("outV", e.get("source", "")))
            in_v = str(e.get("inV", e.get("target", "")))
            result.append((out_v, in_v))
        return result

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for nid in node_ids:
            node = await self.get_node(nid)
            if node is not None:
                result[nid] = node
        return result

    async def get_edges_batch(
        self, pairs: list[tuple[str, str]]
    ) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for src, tgt in pairs:
            edge = await self.get_edge(src, tgt)
            if edge is not None:
                edge_data = {
                    k: v for k, v in edge.items() if k not in ("id", "label", "source", "target")
                }
                result[f"{src}{EDGE_KEY_SEP}{tgt}"] = edge_data
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        result: dict[str, list[tuple[str, str]]] = {}
        for nid in node_ids:
            edges = await self.get_node_edges(nid)
            result[nid] = edges or []
        return result

    # -- write ----------------------------------------------------------------

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        assert self._driver is not None
        props: dict[str, Any] = {}
        for k in ("entity_type", "description", "source_id", "content"):
            if k in node_data:
                props[k] = cap_str(str(node_data[k]))
        await self._driver.upsert_vertex(ENTITY_LABEL, node_id, props)

    async def upsert_nodes_batch(
        self, nodes: list[tuple[str, dict[str, str]]]
    ) -> None:
        for nid, data in nodes:
            await self.upsert_node(nid, data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        assert self._driver is not None
        props: dict[str, Any] = {}
        for k in ("relation_type", "description", "source_id", "content"):
            if k in edge_data:
                props[k] = cap_str(str(edge_data[k]))
        if "weight" in edge_data:
            try:
                props["weight"] = float(edge_data["weight"])
            except (TypeError, ValueError):
                pass
        await self._driver.upsert_edge(
            RELATION_LABEL, source_node_id, target_node_id, props
        )

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        for src, tgt, data in edges:
            await self.upsert_edge(src, tgt, data)

    # -- delete ---------------------------------------------------------------

    async def delete_node(self, node_id: str) -> None:
        assert self._driver is not None
        await self._driver.delete_vertex(node_id)

    async def remove_nodes(self, nodes: list[str]) -> None:
        for nid in nodes:
            await self.delete_node(nid)

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        for src, tgt in edges:
            await self._driver.delete_edges_between(src, tgt, label=RELATION_LABEL)

    # -- embedding ------------------------------------------------------------

    async def embed_node(self, node_id: str) -> Any:
        node = await self.get_node(node_id)
        if node is None:
            return None
        description = node.get("description", "")
        if not description:
            return None
        try:
            embedding = await self.embedding_func([description])
            if isinstance(embedding, list) and embedding:
                return embedding[0]
            return embedding
        except Exception as e:  # noqa: BLE001
            logger.warning(f"embed_node failed for {node_id}: {e}")
            return None

    # -- introspection (best-effort) -----------------------------------------

    async def get_all_nodes(self) -> list[dict]:
        assert self._driver is not None
        verts = await self._driver.list_vertices(label=ENTITY_LABEL, limit=10000)
        return [flatten_vertex(v) for v in verts]

    async def get_all_edges(self) -> list[dict]:
        assert self._driver is not None
        # list_edges_of with empty id lists all edges of that label
        edges = await self._driver.list_edges_of(
            "", direction="BOTH", label=RELATION_LABEL, limit=20000
        ) if False else []
        # Fallback: iterate over all entity vertices and collect their edges.
        verts = await self._driver.list_vertices(label=ENTITY_LABEL, limit=10000)
        edges_set: dict[str, dict] = {}
        for v in verts:
            vid = str(v.get("id", ""))
            if not vid:
                continue
            inc = await self._driver.list_edges_of(
                vid, direction="BOTH", label=RELATION_LABEL, limit=1000
            )
            for e in inc:
                eid = str(e.get("id", ""))
                if eid and eid not in edges_set:
                    edges_set[eid] = flatten_edge(e)
        return list(edges_set.values())

    async def get_all_labels(self) -> list[str]:
        assert self._driver is not None
        verts = await self._driver.list_vertices(label=ENTITY_LABEL, limit=10000)
        labels = {
            v.get("properties", {}).get("entity_type")
            for v in verts
            if v.get("properties", {}).get("entity_type")
        }
        return [str(l) for l in labels if l is not None]

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        verts = await self._driver.list_vertices(label=ENTITY_LABEL, limit=100000)
        counts: dict[str, int] = {}
        for v in verts:
            et = v.get("properties", {}).get("entity_type")
            if et:
                counts[str(et)] = counts.get(str(et), 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [k for k, _ in ranked]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        # No native substring search on secondary index; fall back to in-memory filter.
        verts = await self._driver.list_vertices(label=ENTITY_LABEL, limit=100000)
        q = str(query).lower()
        labels = {
            str(v.get("properties", {}).get("entity_type"))
            for v in verts
            if v.get("properties", {}).get("entity_type")
            and q in str(v.get("properties", {}).get("entity_type")).lower()
        }
        return list(labels)[:limit]

    async def get_knowledge_graph(
        self, node_id: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> dict[str, Any]:
        """Simplified subgraph extraction: k-hop BFS via REST edge listing.

        The webui passes the entity_type (e.g. "artifact") as the label
        parameter; the graph_routes layer routes this to node_id here. We
        detect this case and seed the BFS from all nodes whose entity_type
        matches, instead of from a single (non-existent) node.
        """
        assert self._driver is not None
        # Check if node_id is actually a node id; if not, treat as entity_type
        seed_ids: list[str] = []
        direct = await self._driver.get_vertex(node_id)
        if direct is not None and direct.get("label") == ENTITY_LABEL:
            seed_ids = [node_id]
        else:
            # Treat as entity_type — collect matching node ids as seeds
            type_verts = await self._driver.list_vertices(
                label=ENTITY_LABEL, limit=max_nodes,
                properties={"entity_type": node_id},
            )
            seed_ids = [str(v.get("id")) for v in type_verts if v.get("id")]

        if not seed_ids:
            return {"nodes": {}, "edges": []}

        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        frontier = seed_ids
        visited: set[str] = set()
        for _depth in range(max(1, max_depth)):
            if not frontier or len(nodes) >= max_nodes:
                break
            next_frontier: list[str] = []
            batch = await self.get_nodes_batch(frontier)
            for nid, props in batch.items():
                if nid not in visited:
                    nodes[nid] = props
                    visited.add(nid)
            for nid in frontier:
                edge_list = await self.get_node_edges(nid)
                for src, tgt in edge_list or []:
                    edges.append({"src": src, "tgt": tgt})
                    for other in (src, tgt):
                        if other not in visited and other not in next_frontier:
                            next_frontier.append(other)
            frontier = next_frontier[: max(0, max_nodes - len(nodes))]
        return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# KV storage (generic key/value namespaces)
# ---------------------------------------------------------------------------


@dataclass
class HugeGraphKVStorage(BaseKVStorage):
    """Key/value storage simulated via one HugeGraph vertex label per namespace.

    Each key becomes a vertex (id == key, property ``kv_value`` ==
    JSON-serialized value). The vertex label is created lazily on first use.
    """

    def __init__(
        self,
        namespace: str,
        global_config: dict,
        workspace: str | None = None,
        embedding_func=None,
    ) -> None:
        ws_env = os.environ.get("HUGEGRAPH_WORKSPACE")
        if ws_env and ws_env.strip():
            workspace = ws_env
        if not workspace or not str(workspace).strip():
            workspace = "base"
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver: HugeGraphClient | None = None
        self._kv_label: str | None = None

    async def initialize(self) -> None:
        self._driver = await get_client(self.workspace)
        self._kv_label = await self._driver.ensure_kv_label(self.namespace)

    async def finalize(self) -> None:
        self._driver = None
        self._kv_label = None

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        try:
            assert self._driver is not None and self._kv_label is not None
            verts = await self._driver.list_vertices(label=self._kv_label, limit=100000)
            for v in verts:
                vid = v.get("id")
                if vid is not None:
                    await self._driver.delete_vertex(str(vid))
            return {"status": "success", "message": "data dropped"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": str(e)}

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        assert self._driver is not None and self._kv_label is not None
        v = await self._driver.get_vertex_by_label(id, self._kv_label)
        if v is None:
            return None
        raw = v.get("properties", {}).get("kv_value")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for kid in ids:
            v = await self.get_by_id(kid)
            if v is not None:
                result.append(v)
        return result

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        missing: set[str] = set()
        for k in keys:
            if not await self.has_key(k):
                missing.add(k)
        return missing

    async def has_key(self, key: str) -> bool:
        assert self._driver is not None and self._kv_label is not None
        v = await self._driver.get_vertex_by_label(key, self._kv_label)
        return v is not None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        assert self._driver is not None and self._kv_label is not None
        for key, value in data.items():
            payload = json.dumps(value, ensure_ascii=False, default=str)
            payload_capped = cap_str(payload, 32000)
            await self._driver.upsert_vertex(
                self._kv_label, key, {"kv_value": payload_capped}
            )

    async def delete(self, ids: list[str]) -> None:
        for kid in ids:
            try:
                await self._driver.delete_vertex(kid, label=self._kv_label)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"delete key {kid} failed: {e}")

    async def is_empty(self) -> bool:
        assert self._driver is not None and self._kv_label is not None
        count = await self._driver.count_vertices(label=self._kv_label)
        return count == 0


# ---------------------------------------------------------------------------
# Doc status storage
# ---------------------------------------------------------------------------


@dataclass
class HugeGraphDocStatusStorage(DocStatusStorage):
    """Document processing status backed by the ``lightrag_doc_status`` label."""

    # HG 1.7.0 POST /vertices updates label if vertex id already exists.
    # full_docs KV and doc_status use the same doc_id as key, so the second
    # POST would overwrite the first vertex's label. Prefix doc_status vertex
    # ids with "ds_" to avoid collision.
    @staticmethod
    def _vid(doc_id: str) -> str:
        return f"ds_{doc_id}"

    @staticmethod
    def _doc_id(vid: str) -> str:
        return vid[3:] if vid.startswith("ds_") else vid

    def __init__(
        self,
        namespace: str,
        global_config: dict,
        workspace: str | None = None,
        embedding_func=None,
    ) -> None:
        ws_env = os.environ.get("HUGEGRAPH_WORKSPACE")
        if ws_env and ws_env.strip():
            workspace = ws_env
        if not workspace or not str(workspace).strip():
            workspace = "base"
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver: HugeGraphClient | None = None

    async def initialize(self) -> None:
        self._driver = await get_client(self.workspace)

    async def finalize(self) -> None:
        self._driver = None

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        try:
            assert self._driver is not None
            verts = await self._driver.list_vertices(label=DOC_STATUS_LABEL, limit=100000)
            for v in verts:
                vid = v.get("id")
                if vid is not None:
                    await self._driver.delete_vertex(str(vid))
            return {"status": "success", "message": "data dropped"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": str(e)}

    async def _parse_doc(self, v: dict) -> DocProcessingStatus | None:
        raw = v.get("properties", {}).get("kv_value")
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        # Manually construct DocProcessingStatus (no from_dict classmethod exists)
        try:
            status_raw = data.get("status", "")
            if hasattr(status_raw, "value"):
                status = status_raw
            elif isinstance(status_raw, str):
                # Accept both "PENDING" and "DocStatus.PENDING"
                clean = status_raw.split(".")[-1] if "." in status_raw else status_raw
                try:
                    status = DocStatus(clean)
                except ValueError:
                    status = DocStatus.PENDING
            else:
                status = DocStatus.PENDING
            return DocProcessingStatus(
                content_summary=data.get("content_summary", ""),
                content_length=int(data.get("content_length", 0) or 0),
                file_path=data.get("file_path", data.get("file_basename", "unknown_source")),
                status=status,
                created_at=str(data.get("created_at", "")),
                updated_at=str(data.get("updated_at", "")),
                track_id=data.get("track_id"),
                chunks_count=data.get("chunk_count") or data.get("chunks_count"),
                chunks_list=data.get("chunks_list", []),
                error_msg=data.get("error_msg"),
                metadata=data.get("metadata", {}),
                content_hash=data.get("content_hash"),
            )
        except Exception:
            return None

    async def _fetch_docs_by_statuses(self, statuses: list[str]) -> dict[str, DocProcessingStatus]:
        assert self._driver is not None
        result: dict[str, DocProcessingStatus] = {}
        for st in statuses:
            verts = await self._driver.list_vertices(
                label=DOC_STATUS_LABEL, limit=100000, properties={"doc_status": st}
            )
            for v in verts:
                vid = str(v.get("id", ""))
                if not vid:
                    continue
                # Strip ds_ prefix to get the original doc_id
                doc_id = self._doc_id(vid)
                parsed = await self._parse_doc(v)
                if parsed is not None:
                    result[doc_id] = parsed
        return result

    # -- BaseKVStorage passthroughs ------------------------------------------

    async def get_by_id(self, id: str) -> DocProcessingStatus | None:
        assert self._driver is not None
        v = await self._driver.get_vertex_by_label(self._vid(id), DOC_STATUS_LABEL)
        if v is None:
            return None
        return await self._parse_doc(v)

    async def get_by_ids(self, ids: list[str]) -> list[DocProcessingStatus]:
        result: list[DocProcessingStatus] = []
        for kid in ids:
            v = await self.get_by_id(kid)
            if v is not None:
                result.append(v)
        return result

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        missing: set[str] = set()
        for k in keys:
            v = await self._driver.get_vertex_by_label(self._vid(k), DOC_STATUS_LABEL)
            if v is None:
                missing.add(k)
        return missing

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        assert self._driver is not None
        for doc_id, value in data.items():
            payload = json.dumps(value, ensure_ascii=False, default=str)
            payload_capped = cap_str(payload, 32000)
            if isinstance(value, dict):
                # status may be a DocStatus enum; use .value if available
                status_raw = value.get("status", "")
                if hasattr(status_raw, "value"):
                    status_val = status_raw.value
                else:
                    status_val = str(status_raw)
                content_hash = str(value.get("content_hash", ""))
                file_path = str(value.get("file_path", ""))
                file_basename = str(value.get("file_basename", ""))
                track_id = str(value.get("track_id", ""))
                # LightRAG may pass created_at/updated_at as ISO timestamp
                # strings (e.g. "2026-07-15T09:57:28+00:00") instead of ints.
                # Convert to unix timestamp; fall back to 0 on failure.
                created_at = _to_timestamp(value.get("created_at"))
                updated_at = _to_timestamp(value.get("updated_at"))
                chunk_count = int(value.get("chunk_count") or 0)
                content_length = int(value.get("content_length") or 0)
            else:
                status_val = content_hash = file_path = file_basename = track_id = ""
                created_at = updated_at = chunk_count = content_length = 0

            props = {
                "kv_value": payload_capped,
                "doc_status": status_val or None,
                "content_hash": content_hash or None,
                "file_path": file_path or None,
                "file_basename": file_basename or None,
                "track_id": track_id or None,
                "created_at": created_at or None,
                "updated_at": updated_at or None,
                "chunk_count": chunk_count or None,
                "content_length": content_length or None,
            }
            await self._driver.upsert_vertex(DOC_STATUS_LABEL, self._vid(doc_id), props)

    async def delete(self, ids: list[str]) -> None:
        for kid in ids:
            try:
                await self._driver.delete_vertex(self._vid(kid), label=DOC_STATUS_LABEL)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"delete doc {kid} failed: {e}")

    async def is_empty(self) -> bool:
        count = await self._driver.count_vertices(label=DOC_STATUS_LABEL)
        return count == 0

    # -- DocStatusStorage abstract API ---------------------------------------

    async def get_status_counts(self) -> dict[str, int]:
        assert self._driver is not None
        verts = await self._driver.list_vertices(label=DOC_STATUS_LABEL, limit=100000)
        counts: dict[str, int] = {}
        for v in verts:
            st = v.get("properties", {}).get("doc_status")
            if st:
                counts[str(st)] = counts.get(str(st), 0) + 1
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        return await self._fetch_docs_by_statuses([status.value])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, DocProcessingStatus]:
        return await self._fetch_docs_by_statuses([s.value for s in statuses])

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        assert self._driver is not None
        verts = await self._driver.list_vertices(
            label=DOC_STATUS_LABEL, limit=100000, properties={"track_id": track_id}
        )
        result: dict[str, DocProcessingStatus] = {}
        for v in verts:
            vid = str(v.get("id", ""))
            if not vid:
                continue
            parsed = await self._parse_doc(v)
            if parsed is not None:
                result[vid] = parsed
        return result

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        assert self._driver is not None
        filter_values = self.resolve_status_filter_values(status_filter, status_filters)

        # Fetch matching docs (in-memory paginate since REST has no native ORDER BY)
        if filter_values:
            docs = await self._fetch_docs_by_statuses(list(filter_values))
        else:
            verts = await self._driver.list_vertices(label=DOC_STATUS_LABEL, limit=100000)
            docs: dict[str, DocProcessingStatus] = {}
            for v in verts:
                vid = str(v.get("id", ""))
                if not vid:
                    continue
                doc_id = self._doc_id(vid)
                parsed = await self._parse_doc(v)
                if parsed is not None:
                    docs[doc_id] = parsed

        total = len(docs)
        safe_sort = sort_field if sort_field in ("created_at", "updated_at", "id") else "updated_at"
        reverse = sort_direction == "desc"

        def sort_key(item):
            _vid, dps = item
            if safe_sort == "id":
                return str(_vid)
            val = getattr(dps, safe_sort, "") if not isinstance(dps, dict) else dps.get(safe_sort, "")
            # Normalize to string for consistent sorting (timestamps may be
            # ISO strings or ints depending on write path)
            return str(val) if val is not None else ""

        items = sorted(docs.items(), key=sort_key, reverse=reverse)
        page = max(1, page)
        page_size = max(1, min(200, page_size))
        offset = (page - 1) * page_size
        page_items = items[offset : offset + page_size]
        return page_items, total

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        assert self._driver is not None
        verts = await self._driver.list_vertices(
            label=DOC_STATUS_LABEL, limit=1, properties={"file_path": file_path}
        )
        if not verts:
            return None
        raw = verts[0].get("properties", {}).get("kv_value")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        assert self._driver is not None
        verts = await self._driver.list_vertices(
            label=DOC_STATUS_LABEL, limit=1, properties={"file_basename": basename}
        )
        if not verts:
            return None
        v = verts[0]
        doc_id = self._doc_id(str(v.get("id", "")))
        raw = v.get("properties", {}).get("kv_value")
        if not doc_id or raw is None:
            return None
        try:
            return doc_id, json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:
        assert self._driver is not None
        verts = await self._driver.list_vertices(
            label=DOC_STATUS_LABEL, limit=1, properties={"content_hash": content_hash}
        )
        if not verts:
            return None
        v = verts[0]
        vid = str(v.get("id", ""))
        # Strip ds_ prefix to return the original doc_id
        doc_id = self._doc_id(vid)
        raw = v.get("properties", {}).get("kv_value")
        if not doc_id or raw is None:
            return None
        try:
            return doc_id, json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
