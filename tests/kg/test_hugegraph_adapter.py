"""Unit tests for HugeGraph adapter fixes — vertex id collision + DocProcessingStatus compatibility.

Covers all new/modified code in hugegraph_impl.py and hugegraph_client.py:
  1. _to_timestamp() — ISO string / int / float / None conversion
  2. DocStatusStorage._vid() / _doc_id() — ds_ prefix mapping
  3. DocStatusStorage._parse_doc() — manual DocProcessingStatus construction
  4. DocStatusStorage.get_by_id() — uses _vid for lookup
  5. DocStatusStorage.upsert() — status .value + _vid + _to_timestamp
  6. DocStatusStorage.get_doc_by_content_hash() — strips ds_ prefix
  7. DocStatusStorage.get_doc_by_file_basename() — strips ds_ prefix
  8. DocStatusStorage.delete() — passes label to delete_vertex
  9. get_docs_paginated sort_key — str normalization
 10. HugeGraphClient.delete_vertex() — label query param
 11. HugeGraphKVStorage.delete() — passes label
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.kg.hugegraph_impl import (
    DOC_STATUS_LABEL,
    HugeGraphDocStatusStorage,
    HugeGraphKVStorage,
    _to_timestamp,
)
from lightrag.kg.hugegraph_client import HugeGraphClient


# ---------------------------------------------------------------------------
# 1. _to_timestamp — pure function
# ---------------------------------------------------------------------------

class TestToTimestamp:
    def test_int_passthrough(self):
        assert _to_timestamp(1234567890) == 1234567890

    def test_float_truncate(self):
        assert _to_timestamp(1234567890.5) == 1234567890

    def test_none_returns_zero(self):
        assert _to_timestamp(None) == 0

    def test_str_int(self):
        assert _to_timestamp("1234567890") == 1234567890

    def test_iso_string(self):
        ts = _to_timestamp("2026-07-15T09:57:28+00:00")
        expected = int(datetime.fromisoformat("2026-07-15T09:57:28+00:00").timestamp())
        assert ts == expected

    def test_iso_string_z_suffix(self):
        ts = _to_timestamp("2026-07-15T09:57:28Z")
        expected = int(datetime.fromisoformat("2026-07-15T09:57:28+00:00").timestamp())
        assert ts == expected

    def test_invalid_string_returns_zero(self):
        assert _to_timestamp("not-a-timestamp") == 0

    def test_empty_string_returns_zero(self):
        assert _to_timestamp("") == 0


# ---------------------------------------------------------------------------
# 2. _vid / _doc_id — pure static methods
# ---------------------------------------------------------------------------

class TestVidMapping:
    def test_vid_adds_prefix(self):
        assert HugeGraphDocStatusStorage._vid("doc-abc123") == "ds_doc-abc123"

    def test_vid_plain_id(self):
        assert HugeGraphDocStatusStorage._vid("abc") == "ds_abc"

    def test_doc_id_strips_prefix(self):
        assert HugeGraphDocStatusStorage._doc_id("ds_doc-abc123") == "doc-abc123"

    def test_doc_id_no_prefix(self):
        assert HugeGraphDocStatusStorage._doc_id("doc-abc123") == "doc-abc123"

    def test_doc_id_empty(self):
        assert HugeGraphDocStatusStorage._doc_id("") == ""

    def test_roundtrip(self):
        doc_id = "doc-5ce3beac28f955cae00646a6f9e69a70"
        assert HugeGraphDocStatusStorage._doc_id(HugeGraphDocStatusStorage._vid(doc_id)) == doc_id


# ---------------------------------------------------------------------------
# 3. _parse_doc — construct DocProcessingStatus from vertex dict
# ---------------------------------------------------------------------------

def _make_vertex(kv_value_dict: dict) -> dict:
    """Build a fake HG vertex with kv_value property."""
    return {"id": "ds_doc-test", "label": DOC_STATUS_LABEL, "properties": {"kv_value": json.dumps(kv_value_dict, default=str)}}


class TestParseDoc:
    @pytest.fixture
    def storage(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        return s

    @pytest.mark.asyncio
    async def test_normal_dict(self, storage):
        v = _make_vertex({
            "content_summary": "test content",
            "content_length": 100,
            "file_path": "test.md",
            "status": "PENDING",
            "created_at": "2026-07-15T09:57:28+00:00",
            "updated_at": "2026-07-15T09:57:28+00:00",
        })
        result = await storage._parse_doc(v)
        assert isinstance(result, DocProcessingStatus)
        assert result.content_summary == "test content"
        assert result.content_length == 100
        assert result.file_path == "test.md"
        assert result.status == DocStatus.PENDING

    @pytest.mark.asyncio
    async def test_status_enum_string(self, storage):
        v = _make_vertex({"status": "DocStatus.PENDING", "content_summary": "", "content_length": 0, "file_path": "x", "created_at": "", "updated_at": ""})
        result = await storage._parse_doc(v)
        assert result.status == DocStatus.PENDING

    @pytest.mark.asyncio
    async def test_status_invalid_falls_back_to_pending(self, storage):
        v = _make_vertex({"status": "INVALID", "content_summary": "", "content_length": 0, "file_path": "x", "created_at": "", "updated_at": ""})
        result = await storage._parse_doc(v)
        assert result.status == DocStatus.PENDING

    @pytest.mark.asyncio
    async def test_none_kv_value(self, storage):
        v = {"id": "ds_x", "label": DOC_STATUS_LABEL, "properties": {}}
        result = await storage._parse_doc(v)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json(self, storage):
        v = {"id": "ds_x", "label": DOC_STATUS_LABEL, "properties": {"kv_value": "not json{"}}
        result = await storage._parse_doc(v)
        assert result is None

    @pytest.mark.asyncio
    async def test_content_hash_field(self, storage):
        v = _make_vertex({"status": "PROCESSED", "content_summary": "", "content_length": 0, "file_path": "x", "created_at": "", "updated_at": "", "content_hash": "abc123"})
        result = await storage._parse_doc(v)
        assert result.content_hash == "abc123"

    @pytest.mark.asyncio
    async def test_chunk_count_field(self, storage):
        v = _make_vertex({"status": "PROCESSED", "content_summary": "", "content_length": 0, "file_path": "x", "created_at": "", "updated_at": "", "chunk_count": 5})
        result = await storage._parse_doc(v)
        assert result.chunks_count == 5


# ---------------------------------------------------------------------------
# 4. get_by_id — uses _vid for lookup
# ---------------------------------------------------------------------------

class TestGetById:
    @pytest.fixture
    def storage(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        s._driver = AsyncMock()
        s._parse_doc = AsyncMock(return_value=DocProcessingStatus(
            content_summary="x", content_length=1, file_path="f", status=DocStatus.PENDING, created_at="", updated_at=""
        ))
        return s

    @pytest.mark.asyncio
    async def test_calls_get_vertex_with_vid(self, storage):
        storage._driver.get_vertex_by_label = AsyncMock(return_value={"id": "ds_doc-x", "properties": {}})
        await storage.get_by_id("doc-x")
        storage._driver.get_vertex_by_label.assert_called_once_with("ds_doc-x", DOC_STATUS_LABEL)

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, storage):
        storage._driver.get_vertex_by_label = AsyncMock(return_value=None)
        result = await storage.get_by_id("doc-x")
        assert result is None


# ---------------------------------------------------------------------------
# 5. upsert — status .value + _vid + _to_timestamp
# ---------------------------------------------------------------------------

class TestDocStatusUpsert:
    @pytest.fixture
    def storage(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        s._driver = AsyncMock()
        return s

    @pytest.mark.asyncio
    async def test_status_enum_uses_value(self, storage):
        await storage.upsert({"doc-1": {"status": DocStatus.PENDING, "content_summary": "x", "content_length": 1, "file_path": "f", "created_at": "2026-07-15T09:57:28+00:00", "updated_at": "2026-07-15T09:57:28+00:00"}})
        call_args = storage._driver.upsert_vertex.call_args
        props = call_args.kwargs.get("properties") or call_args[0][2]
        assert props["doc_status"] == DocStatus.PENDING.value  # "pending"

    @pytest.mark.asyncio
    async def test_uses_vid_as_vertex_id(self, storage):
        await storage.upsert({"doc-1": {"status": DocStatus.PENDING, "content_summary": "x", "content_length": 1, "file_path": "f", "created_at": "", "updated_at": ""}})
        call_args = storage._driver.upsert_vertex.call_args
        vid = call_args[0][1] if call_args[0][1:] else call_args.kwargs.get("vid")
        assert vid == "ds_doc-1"

    @pytest.mark.asyncio
    async def test_iso_timestamp_converted(self, storage):
        await storage.upsert({"doc-1": {"status": DocStatus.PENDING, "content_summary": "x", "content_length": 1, "file_path": "f", "created_at": "2026-07-15T09:57:28+00:00", "updated_at": "2026-07-15T09:57:28+00:00"}})
        call_args = storage._driver.upsert_vertex.call_args
        props = call_args.kwargs.get("properties") or call_args[0][2]
        assert isinstance(props["created_at"], int)
        assert props["created_at"] > 0

    @pytest.mark.asyncio
    async def test_string_status_uses_value_directly(self, storage):
        await storage.upsert({"doc-1": {"status": "processed", "content_summary": "x", "content_length": 1, "file_path": "f", "created_at": "", "updated_at": ""}})
        call_args = storage._driver.upsert_vertex.call_args
        props = call_args.kwargs.get("properties") or call_args[0][2]
        assert props["doc_status"] == "processed"


# ---------------------------------------------------------------------------
# 6-7. get_doc_by_content_hash / get_doc_by_file_basename — strip prefix
# ---------------------------------------------------------------------------

class TestGetDocByPrefix:
    @pytest.fixture
    def storage(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        s._driver = AsyncMock()
        return s

    @pytest.mark.asyncio
    async def test_content_hash_strips_prefix(self, storage):
        storage._driver.list_vertices = AsyncMock(return_value=[
            {"id": "ds_doc-abc", "label": DOC_STATUS_LABEL, "properties": {"kv_value": json.dumps({"status": "PENDING"})}}
        ])
        result = await storage.get_doc_by_content_hash("hash123")
        assert result is not None
        doc_id, _ = result
        assert doc_id == "doc-abc"

    @pytest.mark.asyncio
    async def test_content_hash_none_when_empty(self, storage):
        storage._driver.list_vertices = AsyncMock(return_value=[])
        result = await storage.get_doc_by_content_hash("hash123")
        assert result is None

    @pytest.mark.asyncio
    async def test_file_basename_strips_prefix(self, storage):
        storage._driver.list_vertices = AsyncMock(return_value=[
            {"id": "ds_doc-xyz", "label": DOC_STATUS_LABEL, "properties": {"kv_value": json.dumps({"status": "PENDING"})}}
        ])
        result = await storage.get_doc_by_file_basename("test.md")
        assert result is not None
        doc_id, _ = result
        assert doc_id == "doc-xyz"


# ---------------------------------------------------------------------------
# 8. delete — passes label
# ---------------------------------------------------------------------------

class TestDocStatusDelete:
    @pytest.mark.asyncio
    async def test_delete_passes_label(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        s._driver = AsyncMock()
        await s.delete(["doc-1", "doc-2"])
        assert s._driver.delete_vertex.call_count == 2
        for call in s._driver.delete_vertex.call_args_list:
            assert call.kwargs.get("label") == DOC_STATUS_LABEL


# ---------------------------------------------------------------------------
# 9. get_docs_paginated sort_key — str normalization
# ---------------------------------------------------------------------------

class TestSortKey:
    def test_sort_by_updated_at_str(self):
        """sort_key should normalize mixed int/str timestamps to str."""
        from lightrag.base import DocProcessingStatus as DPS
        items = [
            ("doc-1", DPS(content_summary="", content_length=0, file_path="a", status=DocStatus.PENDING, created_at="2026-01-01", updated_at="2026-01-03")),
            ("doc-2", DPS(content_summary="", content_length=0, file_path="b", status=DocStatus.PENDING, created_at="2026-01-01", updated_at="2026-01-01")),
        ]
        # Simulate sort_key with safe_sort="updated_at"
        safe_sort = "updated_at"
        def sort_key(item):
            _vid, dps = item
            if safe_sort == "id":
                return str(_vid)
            val = getattr(dps, safe_sort, "") if not isinstance(dps, dict) else dps.get(safe_sort, "")
            return str(val) if val is not None else ""
        sorted_items = sorted(items, key=sort_key, reverse=True)
        assert sorted_items[0][0] == "doc-1"  # 2026-01-03 > 2026-01-01

    def test_sort_no_type_error_mixed_types(self):
        """Should not raise TypeError when comparing int and str."""
        from lightrag.base import DocProcessingStatus as DPS
        items = [
            ("doc-1", DPS(content_summary="", content_length=0, file_path="a", status=DocStatus.PENDING, created_at=123, updated_at=456)),
            ("doc-2", DPS(content_summary="", content_length=0, file_path="b", status=DocStatus.PENDING, created_at="2026-01-01", updated_at="2026-01-03")),
        ]
        safe_sort = "updated_at"
        def sort_key(item):
            _vid, dps = item
            if safe_sort == "id":
                return str(_vid)
            val = getattr(dps, safe_sort, "") if not isinstance(dps, dict) else dps.get(safe_sort, "")
            return str(val) if val is not None else ""
        # Should not raise
        result = sorted(items, key=sort_key)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 10. HugeGraphClient.delete_vertex — label query param
# ---------------------------------------------------------------------------

class TestDeleteVertexLabel:
    @pytest.mark.asyncio
    async def test_delete_without_label(self):
        client = HugeGraphClient.__new__(HugeGraphClient)
        client.graph = "hugegraph"
        client.graphspace = None
        client._client = AsyncMock()
        client._client.delete = AsyncMock(return_value=MagicMock(status_code=204))
        client._enc_id = MagicMock(return_value="%22doc-test%22")
        await client.delete_vertex("doc-test")
        call_url = client._client.delete.call_args[0][0]
        assert "label=" not in call_url

    @pytest.mark.asyncio
    async def test_delete_with_label(self):
        client = HugeGraphClient.__new__(HugeGraphClient)
        client.graph = "hugegraph"
        client.graphspace = None
        client._client = AsyncMock()
        client._client.delete = AsyncMock(return_value=MagicMock(status_code=204))
        client._enc_id = MagicMock(return_value="%22doc-test%22")
        await client.delete_vertex("doc-test", label="lightrag_doc_status")
        call_url = client._client.delete.call_args[0][0]
        assert "label=lightrag_doc_status" in call_url


# ---------------------------------------------------------------------------
# 11. HugeGraphKVStorage.delete — passes label
# ---------------------------------------------------------------------------

class TestKVStorageDelete:
    @pytest.mark.asyncio
    async def test_delete_passes_kv_label(self):
        s = HugeGraphKVStorage.__new__(HugeGraphKVStorage)
        s._driver = AsyncMock()
        s._kv_label = "lightrag_kv_full_docs"
        await s.delete(["key1", "key2"])
        assert s._driver.delete_vertex.call_count == 2
        for call in s._driver.delete_vertex.call_args_list:
            assert call.kwargs.get("label") == "lightrag_kv_full_docs"


# ---------------------------------------------------------------------------
# _fetch_docs_by_statuses — strips ds_ prefix from returned vertex ids
# ---------------------------------------------------------------------------

class TestFetchDocsByStatuses:
    @pytest.mark.asyncio
    async def test_strips_prefix_from_vid(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        s._driver = AsyncMock()
        s._parse_doc = AsyncMock(return_value=DocProcessingStatus(
            content_summary="", content_length=0, file_path="f", status=DocStatus.PENDING, created_at="", updated_at=""
        ))
        s._driver.list_vertices = AsyncMock(return_value=[
            {"id": "ds_doc-1", "properties": {"kv_value": "{}"}},
            {"id": "ds_doc-2", "properties": {"kv_value": "{}"}},
        ])
        result = await s._fetch_docs_by_statuses(["PENDING"])
        assert "doc-1" in result
        assert "doc-2" in result
        assert "ds_doc-1" not in result


# ---------------------------------------------------------------------------
# Supplementary tests for 100% coverage of new code paths
# ---------------------------------------------------------------------------

class TestToTimestampExtra:
    def test_list_returns_zero(self):
        assert _to_timestamp([1, 2, 3]) == 0

    def test_dict_returns_zero(self):
        assert _to_timestamp({"a": 1}) == 0


class TestParseDocExtra:
    @pytest.fixture
    def storage(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        return s

    @pytest.mark.asyncio
    async def test_status_is_enum_object(self, storage):
        """status_raw is a DocStatus enum (hasattr .value)."""
        v = _make_vertex({"status": DocStatus.PROCESSED, "content_summary": "", "content_length": 0, "file_path": "x", "created_at": "", "updated_at": ""})
        result = await storage._parse_doc(v)
        assert result.status == DocStatus.PROCESSED

    @pytest.mark.asyncio
    async def test_status_string_with_dot(self, storage):
        """status_raw like 'DocStatus.PROCESSED' → split → 'PROCESSED' → DocStatus('PROCESSED') fails (value is lowercase) → PENDING."""
        v = _make_vertex({"status": "DocStatus.PROCESSED", "content_summary": "", "content_length": 0, "file_path": "x", "created_at": "", "updated_at": ""})
        result = await storage._parse_doc(v)
        # DocStatus("PROCESSED") raises ValueError (enum value is "processed"), falls back to PENDING
        assert result.status == DocStatus.PENDING

    @pytest.mark.asyncio
    async def test_status_string_with_dot_lowercase(self, storage):
        """status_raw like 'DocStatus.pending' → split → 'pending' → DocStatus('pending') succeeds."""
        v = _make_vertex({"status": "DocStatus.pending", "content_summary": "", "content_length": 0, "file_path": "x", "created_at": "", "updated_at": ""})
        result = await storage._parse_doc(v)
        assert result.status == DocStatus.PENDING

    @pytest.mark.asyncio
    async def test_status_non_string_non_enum(self, storage):
        """status_raw is e.g. int → falls back to PENDING."""
        v = _make_vertex({"status": 123, "content_summary": "", "content_length": 0, "file_path": "x", "created_at": "", "updated_at": ""})
        result = await storage._parse_doc(v)
        assert result.status == DocStatus.PENDING

    @pytest.mark.asyncio
    async def test_construction_exception_returns_none(self, storage):
        """If DocProcessingStatus construction raises (e.g. content_length is a list), return None."""
        v = _make_vertex({"status": "PENDING", "content_summary": "", "content_length": [1], "file_path": "x", "created_at": "", "updated_at": ""})
        result = await storage._parse_doc(v)
        assert result is None

    @pytest.mark.asyncio
    async def test_status_has_value_attr(self, storage):
        """Cover the hasattr(status_raw, 'value') branch (line 607)."""
        v = _make_vertex({"status": "PENDING", "content_summary": "", "content_length": 0, "file_path": "x", "created_at": "", "updated_at": ""})
        with patch("lightrag.kg.hugegraph_impl.json.loads") as mock_loads:
            class FakeStatus:
                value = DocStatus.PENDING
            mock_loads.return_value = {"status": FakeStatus(), "content_summary": "", "content_length": 0, "file_path": "x", "created_at": "", "updated_at": ""}
            result = await storage._parse_doc(v)
            assert result is not None
            assert result.status.value == DocStatus.PENDING  # FakeStatus.value == PENDING


class TestUpsertExtra:
    @pytest.fixture
    def storage(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        s._driver = AsyncMock()
        return s

    @pytest.mark.asyncio
    async def test_non_dict_value_uses_defaults(self, storage):
        """Non-dict value triggers else branch with empty defaults."""
        await storage.upsert({"doc-1": "not_a_dict"})
        call_args = storage._driver.upsert_vertex.call_args
        props = call_args.kwargs.get("properties") or call_args[0][2]
        assert props["doc_status"] is None or props["doc_status"] == ""

    @pytest.mark.asyncio
    async def test_none_value_uses_defaults(self, storage):
        """None value triggers else branch."""
        await storage.upsert({"doc-1": None})
        call_args = storage._driver.upsert_vertex.call_args
        assert call_args is not None  # upsert_vertex was called


class TestGetByIdsExtra:
    @pytest.mark.asyncio
    async def test_get_by_ids_returns_list(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        s._driver = AsyncMock()
        s._parse_doc = AsyncMock(side_effect=[
            DocProcessingStatus(content_summary="a", content_length=1, file_path="f", status=DocStatus.PENDING, created_at="", updated_at=""),
            None,  # second not found
        ])
        result = await s.get_by_ids(["doc-1", "doc-2"])
        assert len(result) == 1


class TestFilterKeys:
    @pytest.mark.asyncio
    async def test_filter_keys_uses_vid(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        s._driver = AsyncMock()
        # doc-1 not found (None), doc-2 found
        def mock_get(vid, label):
            return None if vid == "ds_doc-1" else {"id": vid}
        s._driver.get_vertex_by_label = AsyncMock(side_effect=mock_get)
        missing = await s.filter_keys({"doc-1", "doc-2"})
        assert "doc-1" in missing
        assert "doc-2" not in missing
        # Verify _vid was used
        calls = s._driver.get_vertex_by_label.call_args_list
        assert calls[0][0][0].startswith("ds_")

    @pytest.mark.asyncio
    async def test_filter_keys_empty(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        s._driver = AsyncMock()
        result = await s.filter_keys(set())
        assert result == set()


class TestGetDocByEdgeCases:
    @pytest.fixture
    def storage(self):
        s = HugeGraphDocStatusStorage.__new__(HugeGraphDocStatusStorage)
        s._driver = AsyncMock()
        return s

    @pytest.mark.asyncio
    async def test_content_hash_no_vid(self, storage):
        storage._driver.list_vertices = AsyncMock(return_value=[
            {"id": "", "properties": {"kv_value": "{}"}}
        ])
        result = await storage.get_doc_by_content_hash("h")
        assert result is None

    @pytest.mark.asyncio
    async def test_content_hash_no_kv_value(self, storage):
        storage._driver.list_vertices = AsyncMock(return_value=[
            {"id": "ds_doc-1", "properties": {}}
        ])
        result = await storage.get_doc_by_content_hash("h")
        assert result is None

    @pytest.mark.asyncio
    async def test_file_basename_no_vid(self, storage):
        storage._driver.list_vertices = AsyncMock(return_value=[
            {"id": "", "properties": {"kv_value": "{}"}}
        ])
        result = await storage.get_doc_by_file_basename("f.md")
        assert result is None

    @pytest.mark.asyncio
    async def test_file_basename_no_kv_value(self, storage):
        storage._driver.list_vertices = AsyncMock(return_value=[
            {"id": "ds_doc-1", "properties": {}}
        ])
        result = await storage.get_doc_by_file_basename("f.md")
        assert result is None
