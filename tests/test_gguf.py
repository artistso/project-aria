"""Tests for reading a GGUF header without a runtime."""

from __future__ import annotations

import struct

import pytest

from aria.llm.gguf import GGUFError, inspect_gguf, looks_like_lfs_pointer


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _build(path, tensors, metadata, version: int = 3):
    data = b"GGUF" + struct.pack("<I", version)
    data += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata))
    for key, kind, payload in metadata:
        data += _string(key) + struct.pack("<I", kind) + payload
    for name, dimensions, type_id in tensors:
        data += _string(name) + struct.pack("<I", len(dimensions))
        data += b"".join(struct.pack("<Q", dim) for dim in dimensions)
        data += struct.pack("<I", type_id) + struct.pack("<Q", 0)
    path.write_bytes(data)
    return path


def test_a_tiny_gguf_is_read(tmp_path) -> None:
    path = _build(
        tmp_path / "tiny.gguf",
        tensors=[("blk.0.attn_q.weight", [4, 8], 12), ("blk.0.ffn.weight", [8, 4], 12)],
        metadata=[
            ("general.architecture", 8, _string("qwen3")),
            ("general.name", 8, _string("tiny-test")),
            ("qwen3.context_length", 4, struct.pack("<I", 4096)),
        ],
    )
    info = inspect_gguf(path)
    assert info.version == 3
    assert info.tensor_count == 2
    assert info.metadata["general.architecture"] == "qwen3"
    assert info.parameters == 64
    assert info.quantisations == {"q4_k": 2}

    summary = info.summary()
    assert summary["parameters_human"] == "64"
    assert summary["dominant_quantisation"] == "q4_k"
    assert summary["context_length"] == 4096


def test_an_lfs_pointer_is_caught_and_named(tmp_path) -> None:
    path = tmp_path / "model.gguf"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\nsize 4915210368\n"
    )
    assert looks_like_lfs_pointer(path) is True
    with pytest.raises(GGUFError) as error:
        inspect_gguf(path)
    assert "LFS pointer" in str(error.value)


def test_a_random_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"\x00\x01\x02\x03" * 16)
    with pytest.raises(GGUFError):
        inspect_gguf(path)


def test_a_truncated_header_is_an_error(tmp_path) -> None:
    path = _build(tmp_path / "cut.gguf", tensors=[("x", [2, 2], 0)], metadata=[])
    path.write_bytes(path.read_bytes()[:-6])
    with pytest.raises((GGUFError, EOFError, struct.error)):
        info = inspect_gguf(path)
        raise GGUFError(f"truncated file still claimed {info.tensor_count} tensors")
