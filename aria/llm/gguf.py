"""Read a GGUF header without a GPU, a runtime, or a download.

The failure this prevents: a ``git clone`` of a Hugging Face repo yields **LFS
pointers** -- 130-byte text files that look like model weights -- and a 4-bit
quantised model cannot be fine-tuned at all. So before anything is trained or
loaded, the file itself has to be inspected.

GGUF is a small, well-defined container: a magic number, a version, then key/value
metadata followed by tensor descriptions. Everything here is read straight out of
the header, which is enough to answer the questions that matter:

* is this a GGUF at all, or an LFS pointer?
* how many parameters does it actually contain?
* what quantisation is the bulk of it?
* which architecture and context length does it declare?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple

__all__ = ["GGUFError", "inspect_gguf", "LFS_POINTER_MAGIC"]

LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/"

MAGIC = b"GGUF"

#: GGML tensor type ids, as used in the header.
TENSOR_TYPES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k",
    14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
    19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs",
    24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64", 30: "bf16",
}

_METADATA_TYPE = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32",
    6: "float32", 7: "bool", 8: "string", 9: "array", 10: "uint64", 11: "int64",
    12: "float64",
}

_SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
    5: ("<i", 4), 6: ("<f", 4), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


class GGUFError(ValueError):
    """The file is not a readable GGUF."""


@dataclass
class GGUFInfo:
    path: str
    version: int
    tensor_count: int
    metadata: Dict[str, object] = field(default_factory=dict)
    tensors: List[dict] = field(default_factory=list)

    @property
    def parameters(self) -> int:
        total = 0
        for tensor in self.tensors:
            count = 1
            for dim in tensor["dimensions"]:
                count *= int(dim)
            total += count
        return total

    @property
    def quantisations(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for tensor in self.tensors:
            name = tensor["type_name"]
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def summary(self) -> dict:
        dominant = next(iter(self.quantisations), None)
        return {
            "path": self.path,
            "format": "gguf",
            "version": self.version,
            "architecture": self.metadata.get("general.architecture"),
            "name": self.metadata.get("general.name"),
            "parameters": self.parameters,
            "parameters_human": _human(self.parameters),
            "tensors": self.tensor_count,
            "quantisations": self.quantisations,
            "dominant_quantisation": dominant,
            "context_length": self.metadata.get("general.context_length")
            or self.metadata.get("llama.context_length")
            or self.metadata.get("qwen2.context_length")
            or self.metadata.get("qwen3.context_length"),
        }


def _human(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    return str(count)


def _read_string(handle: BinaryIO) -> str:
    length = int.from_bytes(handle.read(8), "little")
    return handle.read(length).decode("utf-8", errors="replace")


def _read_value(handle: BinaryIO, kind: int):
    import struct

    if kind == 8:
        return _read_string(handle)
    if kind == 9:
        element_kind = int.from_bytes(handle.read(4), "little")
        length = int.from_bytes(handle.read(8), "little")
        if element_kind == 8:
            return [_read_string(handle) for _ in range(min(length, 8))]
        fmt, size = _SCALAR.get(element_kind, ("<B", 1))
        raw = handle.read(size * length)
        return list(struct.unpack(f"<{length}{fmt[1]}", raw))[:8]
    if kind == 7:
        return bool(int.from_bytes(handle.read(1), "little"))
    if kind in _SCALAR:
        fmt, size = _SCALAR[kind]
        return struct.unpack(fmt, handle.read(size))[0]
    raise GGUFError(f"unknown metadata value type {kind}")


def inspect_gguf(path: str | Path) -> GGUFInfo:
    """Parse the header of a GGUF file. Raises :class:`GGUFError` otherwise."""
    target = Path(path)
    with target.open("rb") as handle:
        head = handle.read(4)
        if head == LFS_POINTER_MAGIC[:4] or head == b"vers":
            handle.seek(0)
            preview = handle.read(200).decode("utf-8", errors="replace")
            raise GGUFError(
                "this is a Git LFS pointer, not a model: "
                + preview.splitlines()[0]
                + " -- fetch the real bytes (huggingface-cli download, not git clone)"
            )
        if head != MAGIC:
            raise GGUFError(
                f"not a GGUF file: magic is {head!r}. If this came from a git clone of a "
                "Hugging Face repo, it is probably an LFS pointer."
            )

        version = int.from_bytes(handle.read(4), "little")
        tensor_count = int.from_bytes(handle.read(8), "little")
        metadata_count = int.from_bytes(handle.read(8), "little")

        metadata: Dict[str, object] = {}
        for _ in range(metadata_count):
            key = _read_string(handle)
            kind = int.from_bytes(handle.read(4), "little")
            metadata[key] = _read_value(handle, kind)

        tensors: List[dict] = []
        for _ in range(tensor_count):
            name = _read_string(handle)
            dimensions_count = int.from_bytes(handle.read(4), "little")
            dimensions = [int.from_bytes(handle.read(8), "little") for _ in range(dimensions_count)]
            type_id = int.from_bytes(handle.read(4), "little")
            offset = int.from_bytes(handle.read(8), "little")  # byte offset of the data
            tensors.append(
                {
                    "name": name,
                    "dimensions": dimensions,
                    "type": type_id,
                    "type_name": TENSOR_TYPES.get(type_id, f"unknown({type_id})"),
                    "offset": offset,
                }
            )

    return GGUFInfo(
        path=str(target),
        version=version,
        tensor_count=tensor_count,
        metadata=metadata,
        tensors=tensors,
    )


def looks_like_lfs_pointer(path: str | Path) -> bool:
    with Path(path).open("rb") as handle:
        return handle.read(len(LFS_POINTER_MAGIC)) == LFS_POINTER_MAGIC
