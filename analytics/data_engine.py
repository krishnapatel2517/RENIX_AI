"""
RENIX Analytics - Data Engine
=============================

Central data-processing engine for RENIX.

Responsibilities:
- Load and normalize datasets
- Detect data formats
- Convert between common representations
- Filter and sort data
- Aggregate records
- Handle missing values
- Provide a common interface for analytics modules
"""

from __future__ import annotations

import csv
import json
import statistics as _statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


class DataEngine:
    """General-purpose data processing engine."""

    SUPPORTED_EXTENSIONS = {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".txt": "text",
        ".xlsx": "excel",
        ".xls": "excel",
    }

    def __init__(self, data: Any = None) -> None:
        self.data = self._normalize(data) if data is not None else []

    # ========================================================
    # LOAD
    # ========================================================

    def load(self, source: Any) -> Any:
        """Load data from a file path or Python object."""

        if isinstance(source, (str, Path)):
            path = Path(source)

            if not path.exists():
                raise FileNotFoundError(f"Data source not found: {path}")

            if not path.is_file():
                raise ValueError(f"Data source is not a file: {path}")

            file_type = self.detect_format(path)

            if file_type == "csv":
                from .csv_reader import CSVReader

                self.data = CSVReader().read(path)

            elif file_type in {"json", "jsonl"}:
                from .json_reader import JSONReader

                self.data = JSONReader().read(path)

            elif file_type == "excel":
                from .excel_reader import ExcelReader

                self.data = ExcelReader().read(path)

            else:
                self.data = path.read_text(encoding="utf-8")

            return self.data

        self.data = self._normalize(source)
        return self.data

    # ========================================================
    # FORMAT DETECTION
    # ========================================================

    def detect_format(self, path: str | Path) -> str:
        """Detect the supported data format from a filename."""

        extension = Path(path).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported data format: {extension or '<none>'}"
            )

        return self.SUPPORTED_EXTENSIONS[extension]

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize(self, data: Any) -> Any:
        """Normalize common Python data structures."""

        if data is None:
            return []

        if isinstance(data, Mapping):
            return [dict(data)]

        if isinstance(data, tuple):
            return [self._normalize_record(item) for item in data]

        if isinstance(data, list):
            return [self._normalize_record(item) for item in data]

        if isinstance(data, set):
            return [self._normalize_record(item) for item in data]

        return data

    @staticmethod
    def _normalize_record(record: Any) -> Any:
        if isinstance(record, Mapping):
            return dict(record)

        if isinstance(record, tuple):
            return list(record)

        return record

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    def count(self) -> int:
        """Return the number of records."""

        if isinstance(self.data, Sequence) and not isinstance(
            self.data, (str, bytes)
        ):
            return len(self.data)

        return 1 if self.data else 0

    def columns(self) -> list[str]:
        """Return columns from a record-oriented dataset."""

        if not isinstance(self.data, list):
            return []

        columns: list[str] = []

        for record in self.data:
            if isinstance(record, Mapping):
                for key in record:
                    key = str(key)
                    if key not in columns:
                        columns.append(key)

        return columns

    def shape(self) -> tuple[int, int]:
        """Return (rows, columns)."""

        return self.count(), len(self.columns())

    # ========================================================
    # FILTERING
    # ========================================================

    def filter(
        self,
        predicate: Callable[[Any], bool] | None = None,
        **conditions: Any,
    ) -> "DataEngine":
        """
        Filter records.

        Example:
            engine.filter(age=15)

        Or:
            engine.filter(lambda row: row["score"] > 50)
        """

        if not isinstance(self.data, list):
            return DataEngine([])

        result = self.data

        if predicate is not None:
            result = [row for row in result if predicate(row)]

        if conditions:
            result = [
                row
                for row in result
                if isinstance(row, Mapping)
                and all(row.get(key) == value for key, value in conditions.items())
            ]

        return DataEngine(result)

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        columns: Iterable[str] | None = None,
        case_sensitive: bool = False,
    ) -> "DataEngine":
        """Search for text across selected columns."""

        if not isinstance(self.data, list):
            return DataEngine([])

        query_text = str(query)

        if not case_sensitive:
            query_text = query_text.lower()

        selected_columns = list(columns) if columns else self.columns()

        result = []

        for row in self.data:
            if not isinstance(row, Mapping):
                continue

            for column in selected_columns:
                value = row.get(column)

                if value is None:
                    continue

                text = str(value)

                if not case_sensitive:
                    text = text.lower()

                if query_text in text:
                    result.append(row)
                    break

        return DataEngine(result)

    # ========================================================
    # SORTING
    # ========================================================

    def sort(
        self,
        by: str | Sequence[str],
        reverse: bool = False,
    ) -> "DataEngine":
        """Sort records by one or more columns."""

        if not isinstance(self.data, list):
            return DataEngine(self.data)

        columns = [by] if isinstance(by, str) else list(by)

        def sort_key(row: Any) -> tuple:
            if not isinstance(row, Mapping):
                return tuple()

            return tuple(
                (
                    row.get(column) is None,
                    str(row.get(column, "")),
                )
                for column in columns
            )

        return DataEngine(sorted(self.data, key=sort_key, reverse=reverse))

    # ========================================================
    # SELECT
    # ========================================================

    def select(self, columns: Sequence[str]) -> "DataEngine":
        """Return only selected columns."""

        if not isinstance(self.data, list):
            return DataEngine([])

        selected = [
            {
                column: row.get(column)
                for column in columns
            }
            for row in self.data
            if isinstance(row, Mapping)
        ]

        return DataEngine(selected)

    # ========================================================
    # DROP COLUMNS
    # ========================================================

    def drop_columns(self, columns: Sequence[str]) -> "DataEngine":
        """Remove selected columns."""

        remove = set(columns)

        if not isinstance(self.data, list):
            return DataEngine(self.data)

        result = [
            {
                key: value
                for key, value in row.items()
                if key not in remove
            }
            for row in self.data
            if isinstance(row, Mapping)
        ]

        return DataEngine(result)

    # ========================================================
    # MISSING VALUES
    # ========================================================

    def missing_values(self) -> dict[str, int]:
        """Count missing values for each column."""

        counts: dict[str, int] = defaultdict(int)

        for row in self.data if isinstance(self.data, list) else []:
            if not isinstance(row, Mapping):
                continue

            for column in self.columns():
                value = row.get(column)

                if value is None or value == "":
                    counts[column] += 1

        return dict(counts)

    def fill_missing(
        self,
        value: Any = None,
        replacements: Mapping[str, Any] | None = None,
    ) -> "DataEngine":
        """Replace missing values."""

        replacements = replacements or {}

        if not isinstance(self.data, list):
            return DataEngine(self.data)

        result = []

        for row in self.data:
            if not isinstance(row, Mapping):
                result.append(row)
                continue

            updated = dict(row)

            for column in self.columns():
                if updated.get(column) in (None, ""):
                    updated[column] = replacements.get(column, value)

            result.append(updated)

        return DataEngine(result)

    # ========================================================
    # UNIQUE VALUES
    # ========================================================

    def unique(self, column: str) -> list[Any]:
        """Return unique values from a column."""

        values = []

        for row in self.data if isinstance(self.data, list) else []:
            if isinstance(row, Mapping):
                value = row.get(column)

                if value not in values:
                    values.append(value)

        return values

    def value_counts(self, column: str) -> dict[Any, int]:
        """Count occurrences of each value."""

        counter: Counter[Any] = Counter()

        for row in self.data if isinstance(self.data, list) else []:
            if isinstance(row, Mapping):
                counter[row.get(column)] += 1

        return dict(counter)

    # ========================================================
    # NUMERIC DATA
    # ========================================================

    def numeric_values(self, column: str) -> list[float]:
        """Extract numeric values from a column."""

        values: list[float] = []

        for row in self.data if isinstance(self.data, list) else []:
            if not isinstance(row, Mapping):
                continue

            value = row.get(column)

            if value is None or value == "":
                continue

            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue

        return values

    def sum(self, column: str) -> float:
        return float(sum(self.numeric_values(column)))

    def mean(self, column: str) -> float | None:
        values = self.numeric_values(column)
        return _statistics.mean(values) if values else None

    def median(self, column: str) -> float | None:
        values = self.numeric_values(column)
        return _statistics.median(values) if values else None

    def minimum(self, column: str) -> float | None:
        values = self.numeric_values(column)
        return min(values) if values else None

    def maximum(self, column: str) -> float | None:
        values = self.numeric_values(column)
        return max(values) if values else None

    # ========================================================
    # GROUPING
    # ========================================================

    def group_by(self, column: str) -> dict[Any, list[dict[str, Any]]]:
        """Group records by a column."""

        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)

        for row in self.data if isinstance(self.data, list) else []:
            if isinstance(row, Mapping):
                groups[row.get(column)].append(dict(row))

        return dict(groups)

    # ========================================================
    # AGGREGATION
    # ========================================================

    def aggregate(
        self,
        group_column: str,
        value_column: str,
        operation: str = "mean",
    ) -> dict[Any, float | int | None]:
        """Aggregate numeric values grouped by another column."""

        groups = self.group_by(group_column)
        result: dict[Any, float | int | None] = {}

        for key, rows in groups.items():
            values = []

            for row in rows:
                try:
                    values.append(float(row[value_column]))
                except (KeyError, TypeError, ValueError):
                    continue

            if not values:
                result[key] = None
                continue

            operation = operation.lower()

            if operation == "sum":
                result[key] = sum(values)

            elif operation == "mean":
                result[key] = _statistics.mean(values)

            elif operation == "median":
                result[key] = _statistics.median(values)

            elif operation == "min":
                result[key] = min(values)

            elif operation == "max":
                result[key] = max(values)

            elif operation == "count":
                result[key] = len(values)

            else:
                raise ValueError(f"Unsupported aggregation: {operation}")

        return result

    # ========================================================
    # TRANSFORM
    # ========================================================

    def transform(
        self,
        column: str,
        function: Callable[[Any], Any],
    ) -> "DataEngine":
        """Apply a function to every value in a column."""

        if not isinstance(self.data, list):
            return DataEngine(self.data)

        result = []

        for row in self.data:
            if not isinstance(row, Mapping):
                result.append(row)
                continue

            updated = dict(row)

            try:
                updated[column] = function(updated.get(column))
            except Exception:
                updated[column] = updated.get(column)

            result.append(updated)

        return DataEngine(result)

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_list(self) -> list[Any]:
        """Return the dataset as a list."""

        if isinstance(self.data, list):
            return list(self.data)

        return [self.data]

    def to_dict(self) -> dict[str, Any]:
        """Return a simple dictionary representation."""

        return {
            "rows": self.count(),
            "columns": self.columns(),
            "data": self.to_list(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the dataset to JSON."""

        return json.dumps(
            self.to_list(),
            indent=indent,
            ensure_ascii=False,
            default=str,
        )

    def save_json(self, path: str | Path, indent: int = 2) -> Path:
        """Save the dataset as JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        output.write_text(
            self.to_json(indent=indent),
            encoding="utf-8",
        )

        return output

    def save_csv(self, path: str | Path) -> Path:
        """Save record-oriented data as CSV."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            row
            for row in self.data
            if isinstance(row, Mapping)
        ]

        columns = self.columns()

        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=columns,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

        return output

    # ========================================================
    # COPY / CLEAR
    # ========================================================

    def copy(self) -> "DataEngine":
        """Create a new independent engine."""

        return DataEngine(self.to_list())

    def clear(self) -> None:
        """Clear the current dataset."""

        self.data = []

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self) -> dict[str, Any]:
        """Generate a compact dataset summary."""

        result: dict[str, Any] = {
            "rows": self.count(),
            "columns": self.columns(),
            "shape": self.shape(),
            "missing_values": self.missing_values(),
        }

        numeric_summary: dict[str, dict[str, Any]] = {}

        for column in self.columns():
            values = self.numeric_values(column)

            if not values:
                continue

            numeric_summary[column] = {
                "count": len(values),
                "sum": sum(values),
                "mean": _statistics.mean(values),
                "median": _statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }

        result["numeric"] = numeric_summary

        return result

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __len__(self) -> int:
        return self.count()

    def __iter__(self):
        return iter(self.to_list())

    def __repr__(self) -> str:
        rows, columns = self.shape()

        return (
            f"DataEngine(rows={rows}, "
            f"columns={columns})"
        )



