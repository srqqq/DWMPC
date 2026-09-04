from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from mcap.well_known import MessageEncoding, SchemaEncoding
from mcap.writer import CompressionType, Writer


FORMAT_NAME = "quadruped_mpc"
TOPIC = "/quadruped_mpc/sample"
SCHEMA_NAME = "quadruped_mpc.DynamicSample"
METADATA_FIELDS = ("vehicle", "trajectory", "method", "condition")


def _validate_metadata(metadata: Mapping[str, object]) -> dict[str, str]:
    missing = [field for field in METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"missing MCAP metadata fields: {missing}")
    return {field: str(metadata[field]) for field in METADATA_FIELDS}


def _numeric_array(value: object, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"field {field!r} must be numeric, got {array.dtype}")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"field {field!r} contains non-finite values")
    return array


def _prepare_record(
    record: Mapping[str, object],
) -> tuple[dict[str, tuple[int, ...]], dict[str, object]]:
    layout = {}
    payload = {}
    for field, value in record.items():
        if not isinstance(field, str) or not field:
            raise ValueError("snapshot field names must be non-empty strings")
        array = _numeric_array(value, field)
        layout[field] = array.shape
        if field != "t":
            payload[field] = float(array) if array.ndim == 0 else array.tolist()
    return layout, payload


def _payload_schema(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {key: _payload_schema(child) for key, child in value.items()},
            "required": list(value),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        item_schema = _payload_schema(value[0]) if value else {"type": "number"}
        return {
            "type": "array",
            "items": item_schema,
            "minItems": len(value),
            "maxItems": len(value),
        }
    return {"type": "number"}


class MCAPDataLogger:
    """Stream fixed-layout numeric snapshots to one compressed MCAP file."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        vehicle: str,
        trajectory: str,
        method: str,
        condition: str,
    ) -> None:
        self.metadata = _validate_metadata(
            {
                "vehicle": vehicle,
                "trajectory": trajectory,
                "method": method,
                "condition": condition,
            }
        )
        filename = "_".join(
            self.metadata[field] for field in METADATA_FIELDS
        ) + ".mcap"
        self.file_path = Path(output_dir).expanduser().resolve() / filename
        self._stream = None
        self._writer: Writer | None = None
        self._channel_id: int | None = None
        self._layout: dict[str, tuple[int, ...]] | None = None
        self._sequence = 0

    def _open(self, payload: dict[str, object]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.file_path.open("wb")
        self._writer = Writer(self._stream, compression=CompressionType.ZSTD)
        self._writer.start(profile=FORMAT_NAME)
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            **_payload_schema(payload),
        }
        schema_id = self._writer.register_schema(
            name=SCHEMA_NAME,
            encoding=SchemaEncoding.JSONSchema,
            data=json.dumps(schema, separators=(",", ":")).encode(),
        )
        self._channel_id = self._writer.register_channel(
            topic=TOPIC,
            message_encoding=MessageEncoding.JSON,
            schema_id=schema_id,
        )
        self._writer.add_metadata(name=FORMAT_NAME, data=self.metadata)

    def log(self, snapshot: Mapping[str, object]) -> None:
        if "t" not in snapshot:
            raise ValueError("snapshot must contain 't'")

        layout, payload = _prepare_record(snapshot)
        if layout["t"]:
            raise ValueError("snapshot field 't' must be a scalar")
        if self._layout is None:
            self._layout = layout
        elif layout != self._layout:
            raise ValueError(
                f"snapshot layout changed; expected {self._layout}, got {layout}"
            )

        timestamp = float(snapshot["t"])
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(f"t must be finite and non-negative, got {timestamp}")
        if self._writer is None:
            self._open(payload)

        assert self._writer is not None and self._channel_id is not None
        timestamp_ns = round(timestamp * 1_000_000_000)
        self._writer.add_message(
            channel_id=self._channel_id,
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
            sequence=self._sequence,
            data=json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(),
        )
        self._sequence += 1

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.finish()
        finally:
            assert self._stream is not None
            self._stream.close()
            self._writer = None
            self._stream = None
