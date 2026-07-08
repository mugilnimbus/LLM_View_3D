from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Any


class GgufMetadataError(ValueError):
    pass


GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: "B",
    GGUF_TYPE_INT8: "b",
    GGUF_TYPE_UINT16: "H",
    GGUF_TYPE_INT16: "h",
    GGUF_TYPE_UINT32: "I",
    GGUF_TYPE_INT32: "i",
    GGUF_TYPE_FLOAT32: "f",
    GGUF_TYPE_BOOL: "?",
    GGUF_TYPE_UINT64: "Q",
    GGUF_TYPE_INT64: "q",
    GGUF_TYPE_FLOAT64: "d",
}

_SCALAR_SIZES = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}

_GENERAL_KEYS = {
    "general.architecture",
    "general.basename",
    "general.file_type",
    "general.name",
    "general.size_label",
    "general.type",
}

_ARCHITECTURE_SUFFIXES = (
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
    ".block_count",
    ".context_length",
    ".embedding_length",
    ".feed_forward_length",
    ".rope.dimension_count",
    ".rope.freq_base",
)


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Read architecture metadata from a GGUF file without loading tensors."""
    metadata: dict[str, Any] = {}
    try:
        with path.open("rb") as file:
            if file.read(4) != b"GGUF":
                raise GgufMetadataError(f"{path.name} is not a GGUF file")

            _version = _read_struct(file, "I")
            _tensor_count = _read_struct(file, "Q")
            metadata_count = _read_struct(file, "Q")

            for _ in range(metadata_count):
                key = _read_string(file)
                value_type = _read_struct(file, "I")
                if _should_keep(key):
                    metadata[key] = _read_value(file, value_type)
                elif key == "tokenizer.ggml.tokens":
                    metadata["tokenizer.ggml.tokens"] = _skip_array_and_return_count(file)
                else:
                    _skip_value(file, value_type)
    except OSError as exc:
        raise GgufMetadataError(f"Could not read {path}: {exc}") from exc
    return metadata


def _should_keep(key: str) -> bool:
    return key in _GENERAL_KEYS or key.endswith(_ARCHITECTURE_SUFFIXES)


def _read_value(file: BinaryIO, value_type: int) -> Any:
    if value_type == GGUF_TYPE_STRING:
        return _read_string(file)
    if value_type == GGUF_TYPE_ARRAY:
        return _read_array(file)
    if value_type in _SCALAR_FORMATS:
        return _read_struct(file, _SCALAR_FORMATS[value_type])
    raise GgufMetadataError(f"Unsupported GGUF metadata type: {value_type}")


def _read_array(file: BinaryIO) -> list[Any]:
    element_type = _read_struct(file, "I")
    count = _read_struct(file, "Q")
    if count > 512 or element_type == GGUF_TYPE_STRING:
        _skip_array_items(file, element_type, count)
        return []
    return [_read_value(file, element_type) for _ in range(count)]


def _skip_value(file: BinaryIO, value_type: int) -> None:
    if value_type == GGUF_TYPE_STRING:
        file.seek(_read_struct(file, "Q"), 1)
        return
    if value_type == GGUF_TYPE_ARRAY:
        _skip_array_and_return_count(file)
        return
    size = _SCALAR_SIZES.get(value_type)
    if size is None:
        raise GgufMetadataError(f"Unsupported GGUF metadata type: {value_type}")
    file.seek(size, 1)


def _skip_array_and_return_count(file: BinaryIO) -> int:
    element_type = _read_struct(file, "I")
    count = _read_struct(file, "Q")
    _skip_array_items(file, element_type, count)
    return count


def _skip_array_items(file: BinaryIO, element_type: int, count: int) -> None:
    if element_type == GGUF_TYPE_STRING:
        for _ in range(count):
            file.seek(_read_struct(file, "Q"), 1)
        return

    size = _SCALAR_SIZES.get(element_type)
    if size is None:
        for _ in range(count):
            _skip_value(file, element_type)
        return
    file.seek(size * count, 1)


def _read_string(file: BinaryIO) -> str:
    length = _read_struct(file, "Q")
    return file.read(length).decode("utf-8", errors="replace")


def _read_struct(file: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = file.read(size)
    if len(data) != size:
        raise GgufMetadataError("Unexpected end of GGUF metadata")
    return struct.unpack(f"<{fmt}", data)[0]
