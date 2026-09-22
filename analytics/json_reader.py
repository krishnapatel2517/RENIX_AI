"""
RENIX Analytics - JSON Reader
=============================

JSON and JSONL loading/writing utilities for RENIX analytics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class JSONReader:
    """Read, inspect, transform, and write JSON datasets."""

    SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".ndjson"}

    def __init__(
        self,
        encoding: str = "utf-8",
    ) -> None:
        self.encoding = encoding

    # ========================================================
    # READ
    # ========================================================

    def read(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
    ) -> Any:
        """Read a standard JSON file."""

        file_path = self._validate_path(path)

        with file_path.open(
            "r",
            encoding=encoding or self.encoding,
        ) as file:
            return json.load(file)

    # ========================================================
    # READ JSONL
    # ========================================================

    def read_jsonl(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
        ignore_invalid: bool = False,
    ) -> list[Any]:
        """Read newline-delimited JSON."""

        file_path = self._validate_path(path)

        result: list[Any] = []

        with file_path.open(
            "r",
            encoding=encoding or self.encoding,
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    if not ignore_invalid:
                        raise ValueError(
                            f"Invalid JSON on line "
                            f"{line_number} of {file_path}"
                        )

        return result

    # ========================================================
    # AUTO READ
    # ========================================================

    def read_auto(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
    ) -> Any:
        """Automatically select JSON or JSONL based on extension."""

        file_path = self._validate_path(path)

        if file_path.suffix.lower() in {
            ".jsonl",
            ".ndjson",
        }:
            return self.read_jsonl(
                file_path,
                encoding=encoding,
            )

        return self.read(
            file_path,
            encoding=encoding,
        )

    # ========================================================
    # SAMPLE
    # ========================================================

    def sample(
        self,
        path: str | Path,
        count: int = 5,
        *,
        encoding: str | None = None,
    ) -> list[Any]:
        """Return a sample of records from JSON data."""

        if count < 0:
            raise ValueError("count must be >= 0")

        data = self.read_auto(
            path,
            encoding=encoding,
        )

        if isinstance(data, list):
            return data[:count]

        if isinstance(data, Mapping):
            items = list(data.items())[:count]
            return [
                {
                    "key": key,
                    "value": value,
                }
                for key, value in items
            ]

        return [data] if count > 0 else []

    # ========================================================
    # WRITE
    # ========================================================

    def write(
        self,
        path: str | Path,
        data: Any,
        *,
        indent: int = 2,
        ensure_ascii: bool = False,
        encoding: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """Write data as standard JSON."""

        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {output}"
            )

        with output.open(
            "w",
            encoding=encoding or self.encoding,
        ) as file:
            json.dump(
                data,
                file,
                indent=indent,
                ensure_ascii=ensure_ascii,
                default=self._serialize,
            )

        return output

    # ========================================================
    # WRITE JSONL
    # ========================================================

    def write_jsonl(
        self,
        path: str | Path,
        records: Iterable[Any],
        *,
        ensure_ascii: bool = False,
        encoding: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        """Write records as newline-delimited JSON."""

        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {output}"
            )

        with output.open(
            "w",
            encoding=encoding or self.encoding,
        ) as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=ensure_ascii,
                        default=self._serialize,
                    )
                )
                file.write("\n")

        return output

    # ========================================================
    # APPEND JSONL
    # ========================================================

    def append_jsonl(
        self,
        path: str | Path,
        records: Iterable[Any],
        *,
        ensure_ascii: bool = False,
        encoding: str | None = None,
    ) -> Path:
        """Append records to a JSONL file."""

        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "a",
            encoding=encoding or self.encoding,
        ) as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=ensure_ascii,
                        default=self._serialize,
                    )
                )
                file.write("\n")

        return output

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
    ) -> dict[str, Any]:
        """Validate a JSON or JSONL file."""

        file_path = self._validate_path(path)

        try:
            if file_path.suffix.lower() in {
                ".jsonl",
                ".ndjson",
            }:
                records = self.read_jsonl(
                    file_path,
                    encoding=encoding,
                    ignore_invalid=False,
                )

                return {
                    "valid": True,
                    "format": "jsonl",
                    "records": len(records),
                    "errors": [],
                }

            data = self.read(
                file_path,
                encoding=encoding,
            )

            return {
                "valid": True,
                "format": "json",
                "type": type(data).__name__,
                "records": self._record_count(data),
                "errors": [],
            }

        except (
            json.JSONDecodeError,
            ValueError,
            UnicodeDecodeError,
        ) as exc:
            return {
                "valid": False,
                "format": self._detect_format(file_path),
                "records": 0,
                "errors": [str(exc)],
            }

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        path: str | Path,
        query: str,
        *,
        encoding: str | None = None,
        case_sensitive: bool = False,
    ) -> list[Any]:
        """Search recursively through JSON records."""

        data = self.read_auto(
            path,
            encoding=encoding,
        )

        result: list[Any] = []

        needle = str(query)

        if not case_sensitive:
            needle = needle.lower()

        def contains(value: Any) -> bool:
            text = str(value)

            if not case_sensitive:
                text = text.lower()

            return needle in text

        def visit(value: Any) -> bool:
            if isinstance(value, Mapping):
                return any(
                    visit(key) or visit(item)
                    for key, item in value.items()
                )

            if isinstance(value, list):
                return any(
                    visit(item)
                    for item in value
                )

            return contains(value)

        if isinstance(data, list):
            for item in data:
                if visit(item):
                    result.append(item)

        elif visit(data):
            result.append(data)

        return result

    # ========================================================
    # FLATTEN
    # ========================================================

    def flatten(
        self,
        data: Any,
        *,
        separator: str = ".",
    ) -> dict[str, Any]:
        """Flatten nested JSON objects."""

        result: dict[str, Any] = {}

        def visit(
            value: Any,
            prefix: str = "",
        ) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    new_prefix = (
                        f"{prefix}{separator}{key}"
                        if prefix
                        else str(key)
                    )
                    visit(child, new_prefix)

            elif isinstance(value, list):
                for index, child in enumerate(value):
                    new_prefix = (
                        f"{prefix}{separator}{index}"
                        if prefix
                        else str(index)
                    )
                    visit(child, new_prefix)

            else:
                result[prefix] = value

        visit(data)

        return result

    # ========================================================
    # UNFLATTEN
    # ========================================================

    def unflatten(
        self,
        data: Mapping[str, Any],
        *,
        separator: str = ".",
    ) -> dict[str, Any]:
        """Convert flattened keys into a nested dictionary."""

        result: dict[str, Any] = {}

        for key, value in data.items():
            parts = str(key).split(separator)

            current = result

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}

                if not isinstance(
                    current[part],
                    dict,
                ):
                    current[part] = {}

                current = current[part]

            current[parts[-1]] = value

        return result

    # ========================================================
    # EXTRACT
    # ========================================================

    def extract(
        self,
        data: Mapping[str, Any],
        path: str,
        *,
        separator: str = ".",
        default: Any = None,
    ) -> Any:
        """Extract a nested value using a dotted path."""

        current: Any = data

        for part in path.split(separator):
            if isinstance(current, Mapping):
                if part not in current:
                    return default

                current = current[part]

            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (
                    ValueError,
                    IndexError,
                ):
                    return default

            else:
                return default

        return current

    # ========================================================
    # RECORD NORMALIZATION
    # ========================================================

    def records(
        self,
        data: Any,
    ) -> list[dict[str, Any]]:
        """
        Normalize common JSON structures into records.

        Dictionaries become one record.
        Lists of dictionaries remain records.
        Primitive lists become {'value': ...}.
        """

        if data is None:
            return []

        if isinstance(data, Mapping):
            return [dict(data)]

        if isinstance(data, list):
            result = []

            for item in data:
                if isinstance(item, Mapping):
                    result.append(dict(item))
                else:
                    result.append({"value": item})

            return result

        return [{"value": data}]

    # ========================================================
    # TYPE INFORMATION
    # ========================================================

    def structure(
        self,
        data: Any,
    ) -> dict[str, Any]:
        """Return a recursive description of JSON structure."""

        def describe(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    str(key): describe(child)
                    for key, child in value.items()
                }

            if isinstance(value, list):
                if not value:
                    return []

                return [
                    describe(value[0])
                ]

            return type(value).__name__

        return describe(data)

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _validate_path(
        path: str | Path,
    ) -> Path:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"JSON file not found: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"JSON path is not a file: {file_path}"
            )

        if file_path.suffix.lower() not in {
            ".json",
            ".jsonl",
            ".ndjson",
        }:
            raise ValueError(
                f"Unsupported JSON extension: "
                f"{file_path.suffix}"
            )

        return file_path

    @staticmethod
    def _detect_format(
        path: Path,
    ) -> str:
        if path.suffix.lower() in {
            ".jsonl",
            ".ndjson",
        }:
            return "jsonl"

        return "json"

    @staticmethod
    def _record_count(data: Any) -> int:
        if isinstance(data, list):
            return len(data)

        if data is None:
            return 0

        return 1

    @staticmethod
    def _serialize(value: Any) -> Any:
        """Serialize common non-JSON-native values."""

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "isoformat"):
            return value.isoformat()

        if isinstance(value, set):
            return list(value)

        if hasattr(value, "__dict__"):
            return value.__dict__

        return str(value)

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            f"JSONReader(encoding={self.encoding!r})"
        )



