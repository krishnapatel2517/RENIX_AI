"""
RENIX Analytics - Tables
========================

Utilities for creating, transforming, sorting, filtering, and exporting
tabular analytics data.

This module intentionally avoids requiring pandas so the core analytics
layer remains lightweight.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


class TableEngine:
    """General-purpose tabular data engine."""

    def __init__(
        self,
        default_encoding: str = "utf-8-sig",
    ) -> None:
        self.default_encoding = default_encoding

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize(
        rows: Iterable[Mapping[str, Any]],
        *,
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Normalize records into dictionaries with consistent columns.
        """

        records = [
            dict(row)
            for row in rows
        ]

        if columns is None:
            discovered: list[str] = []

            for row in records:
                for key in row:
                    key = str(key)

                    if key not in discovered:
                        discovered.append(key)

            columns = discovered

        column_names = [
            str(column)
            for column in columns
        ]

        return [
            {
                column: row.get(column)
                for column in column_names
            }
            for row in records
        ]

    # ========================================================
    # COLUMNS
    # ========================================================

    def columns(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> list[str]:
        """Return discovered table columns."""

        records = list(rows)
        result: list[str] = []

        for row in records:
            for key in row:
                key = str(key)

                if key not in result:
                    result.append(key)

        return result

    def select_columns(
        self,
        rows: Iterable[Mapping[str, Any]],
        columns: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Select specific columns."""

        selected = [
            str(column)
            for column in columns
        ]

        return [
            {
                column: row.get(column)
                for column in selected
            }
            for row in rows
        ]

    def rename_columns(
        self,
        rows: Iterable[Mapping[str, Any]],
        mapping: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        """Rename columns."""

        return [
            {
                str(mapping.get(
                    key,
                    key,
                )): value
                for key, value in row.items()
            }
            for row in rows
        ]

    # ========================================================
    # ADD / REMOVE
    # ========================================================

    def add_column(
        self,
        rows: Iterable[Mapping[str, Any]],
        column: str,
        values: Iterable[Any] | Any,
    ) -> list[dict[str, Any]]:
        """Add a column using values or one constant value."""

        records = [
            dict(row)
            for row in rows
        ]

        if isinstance(values, Iterable) and not isinstance(
            values,
            (str, bytes, dict),
        ):
            value_list = list(values)

            if len(value_list) != len(records):
                raise ValueError(
                    "Number of values must match number of rows."
                )
        else:
            value_list = [
                values
                for _ in records
            ]

        for row, value in zip(
            records,
            value_list,
        ):
            row[column] = value

        return records

    def remove_columns(
        self,
        rows: Iterable[Mapping[str, Any]],
        columns: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Remove selected columns."""

        remove = {
            str(column)
            for column in columns
        }

        return [
            {
                key: value
                for key, value in row.items()
                if key not in remove
            }
            for row in rows
        ]

    # ========================================================
    # FILTERING
    # ========================================================

    def filter(
        self,
        rows: Iterable[Mapping[str, Any]],
        predicate: Callable[
            [Mapping[str, Any]],
            bool,
        ],
    ) -> list[dict[str, Any]]:
        """Filter rows using a predicate."""

        return [
            dict(row)
            for row in rows
            if predicate(row)
        ]

    def where(
        self,
        rows: Iterable[Mapping[str, Any]],
        column: str,
        value: Any,
    ) -> list[dict[str, Any]]:
        """Filter rows where a column equals a value."""

        return [
            dict(row)
            for row in rows
            if row.get(column) == value
        ]

    def contains(
        self,
        rows: Iterable[Mapping[str, Any]],
        column: str,
        query: str,
        *,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """Filter rows where a column contains text."""

        needle = str(query)

        if not case_sensitive:
            needle = needle.lower()

        result = []

        for row in rows:
            value = row.get(column)

            if value is None:
                continue

            text = str(value)

            if not case_sensitive:
                text = text.lower()

            if needle in text:
                result.append(dict(row))

        return result

    # ========================================================
    # SORTING
    # ========================================================

    def sort(
        self,
        rows: Iterable[Mapping[str, Any]],
        column: str,
        *,
        descending: bool = False,
        missing_last: bool = True,
    ) -> list[dict[str, Any]]:
        """Sort rows by a column."""

        records = [
            dict(row)
            for row in rows
        ]

        if missing_last:
            records.sort(
                key=lambda row: (
                    row.get(column) is None,
                    self._safe_sort_value(
                        row.get(column)
                    ),
                ),
                reverse=descending,
            )
        else:
            records.sort(
                key=lambda row: self._safe_sort_value(
                    row.get(column)
                ),
                reverse=descending,
            )

        return records

    def sort_multiple(
        self,
        rows: Iterable[Mapping[str, Any]],
        specifications: Sequence[
            tuple[str, bool]
        ],
    ) -> list[dict[str, Any]]:
        """
        Sort using multiple columns.

        Each specification is:
            (column_name, descending)
        """

        records = [
            dict(row)
            for row in rows
        ]

        # Stable sorting: apply the last key first.
        for column, descending in reversed(
            specifications
        ):
            records.sort(
                key=lambda row, key=column:
                    self._safe_sort_value(
                        row.get(key)
                    ),
                reverse=descending,
            )

        return records

    # ========================================================
    # LIMIT / OFFSET
    # ========================================================

    def limit(
        self,
        rows: Iterable[Mapping[str, Any]],
        count: int,
    ) -> list[dict[str, Any]]:
        """Return the first N rows."""

        if count < 0:
            raise ValueError(
                "count must be >= 0"
            )

        return [
            dict(row)
            for row in list(rows)[:count]
        ]

    def offset(
        self,
        rows: Iterable[Mapping[str, Any]],
        count: int,
    ) -> list[dict[str, Any]]:
        """Skip the first N rows."""

        if count < 0:
            raise ValueError(
                "count must be >= 0"
            )

        return [
            dict(row)
            for row in list(rows)[count:]
        ]

    def paginate(
        self,
        rows: Iterable[Mapping[str, Any]],
        page: int,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """Return one page of tabular data."""

        if page < 1:
            raise ValueError(
                "page must be >= 1"
            )

        if page_size < 1:
            raise ValueError(
                "page_size must be >= 1"
            )

        records = [
            dict(row)
            for row in rows
        ]

        total = len(records)

        start = (
            page - 1
        ) * page_size

        end = start + page_size

        page_rows = records[start:end]

        return {
            "rows": page_rows,
            "page": page,
            "page_size": page_size,
            "total_rows": total,
            "total_pages": (
                (total + page_size - 1)
                // page_size
                if total
                else 0
            ),
            "has_previous": page > 1,
            "has_next": end < total,
        }

    # ========================================================
    # UNIQUE / DUPLICATES
    # ========================================================

    def unique(
        self,
        rows: Iterable[Mapping[str, Any]],
        column: str,
    ) -> list[Any]:
        """Return unique values from a column."""

        result: list[Any] = []

        for row in rows:
            value = row.get(column)

            if value not in result:
                result.append(value)

        return result

    def duplicates(
        self,
        rows: Iterable[Mapping[str, Any]],
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return rows belonging to duplicate groups."""

        records = [
            dict(row)
            for row in rows
        ]

        if columns is None:
            columns = self.columns(records)

        keys = [
            tuple(
                self._hashable(row.get(column))
                for column in columns
            )
            for row in records
        ]

        counts: dict[Any, int] = {}

        for key in keys:
            counts[key] = counts.get(key, 0) + 1

        return [
            row
            for row, key in zip(
                records,
                keys,
            )
            if counts[key] > 1
        ]

    # ========================================================
    # GROUPING
    # ========================================================

    def group_by(
        self,
        rows: Iterable[Mapping[str, Any]],
        column: str,
    ) -> dict[Any, list[dict[str, Any]]]:
        """Group rows by a column."""

        groups: dict[Any, list[dict[str, Any]]] = {}

        for row in rows:
            value = row.get(column)

            key = self._hashable(value)

            groups.setdefault(
                key,
                [],
            ).append(dict(row))

        return groups

    def group_count(
        self,
        rows: Iterable[Mapping[str, Any]],
        column: str,
    ) -> list[dict[str, Any]]:
        """Count rows in each group."""

        groups = self.group_by(
            rows,
            column,
        )

        return [
            {
                column: key,
                "count": len(group),
            }
            for key, group in groups.items()
        ]

    # ========================================================
    # AGGREGATION
    # ========================================================

    def aggregate(
        self,
        rows: Iterable[Mapping[str, Any]],
        column: str,
        operation: str,
    ) -> Any:
        """Aggregate one numeric column."""

        values = [
            row.get(column)
            for row in rows
            if row.get(column) is not None
        ]

        numbers = []

        for value in values:
            try:
                if isinstance(value, bool):
                    continue

                numbers.append(float(value))

            except (TypeError, ValueError):
                continue

        operation = operation.lower().strip()

        if operation == "count":
            return len(values)

        if not numbers:
            return None

        if operation == "sum":
            return sum(numbers)

        if operation == "mean":
            return sum(numbers) / len(numbers)

        if operation == "min":
            return min(numbers)

        if operation == "max":
            return max(numbers)

        if operation == "range":
            return max(numbers) - min(numbers)

        if operation == "median":
            ordered = sorted(numbers)
            middle = len(ordered) // 2

            if len(ordered) % 2:
                return ordered[middle]

            return (
                ordered[middle - 1]
                + ordered[middle]
            ) / 2

        raise ValueError(
            f"Unsupported aggregation: {operation}"
        )

    # ========================================================
    # PIVOT
    # ========================================================

    def pivot(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        index: str,
        columns: str,
        values: str,
        aggregation: str = "sum",
    ) -> list[dict[str, Any]]:
        """
        Create a simple pivot table.

        Example:

            index       month       sales
            Mumbai      Jan         100
            Mumbai      Feb         150
        """

        records = [
            dict(row)
            for row in rows
        ]

        index_values = self.unique(
            records,
            index,
        )

        column_values = self.unique(
            records,
            columns,
        )

        result: list[dict[str, Any]] = []

        for index_value in index_values:
            output_row: dict[str, Any] = {
                index: index_value
            }

            for column_value in column_values:
                matching = [
                    row
                    for row in records
                    if row.get(index) == index_value
                    and row.get(columns) == column_value
                ]

                aggregate_values = [
                    row.get(values)
                    for row in matching
                ]

                output_row[
                    str(column_value)
                ] = self._aggregate_values(
                    aggregate_values,
                    aggregation,
                )

            result.append(output_row)

        return result

    # ========================================================
    # JOIN
    # ========================================================

    def join(
        self,
        left: Iterable[Mapping[str, Any]],
        right: Iterable[Mapping[str, Any]],
        *,
        left_on: str,
        right_on: str | None = None,
        how: str = "inner",
    ) -> list[dict[str, Any]]:
        """Join two tables."""

        left_rows = [
            dict(row)
            for row in left
        ]

        right_rows = [
            dict(row)
            for row in right
        ]

        right_key = right_on or left_on

        how = how.lower().strip()

        if how not in {
            "inner",
            "left",
            "right",
        }:
            raise ValueError(
                "how must be inner, left, or right"
            )

        right_index: dict[Any, list[dict[str, Any]]] = {}

        for row in right_rows:
            key = self._hashable(
                row.get(right_key)
            )

            right_index.setdefault(
                key,
                [],
            ).append(row)

        result: list[dict[str, Any]] = []

        for left_row in left_rows:
            key = self._hashable(
                left_row.get(left_on)
            )

            matches = right_index.get(
                key,
                [],
            )

            if matches:
                for right_row in matches:
                    result.append(
                        self._merge_rows(
                            left_row,
                            right_row,
                        )
                    )

            elif how == "left":
                result.append(dict(left_row))

        if how == "right":
            left_keys = {
                self._hashable(
                    row.get(left_on)
                )
                for row in left_rows
            }

            for right_row in right_rows:
                key = self._hashable(
                    right_row.get(right_key)
                )

                if key not in left_keys:
                    result.append(dict(right_row))

        return result

    # ========================================================
    # TRANSFORM
    # ========================================================

    def transform_column(
        self,
        rows: Iterable[Mapping[str, Any]],
        column: str,
        function: Callable[[Any], Any],
    ) -> list[dict[str, Any]]:
        """Transform every value in one column."""

        result = []

        for row in rows:
            new_row = dict(row)

            new_row[column] = function(
                row.get(column)
            )

            result.append(new_row)

        return result

    # ========================================================
    # NULL HANDLING
    # ========================================================

    def fill_missing(
        self,
        rows: Iterable[Mapping[str, Any]],
        column: str,
        value: Any,
    ) -> list[dict[str, Any]]:
        """Replace missing values in a column."""

        result = []

        for row in rows:
            new_row = dict(row)

            if (
                column not in new_row
                or new_row[column] is None
                or new_row[column] == ""
            ):
                new_row[column] = value

            result.append(new_row)

        return result

    def drop_missing(
        self,
        rows: Iterable[Mapping[str, Any]],
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Remove rows containing missing values."""

        records = [
            dict(row)
            for row in rows
        ]

        if columns is None:
            columns = self.columns(records)

        return [
            row
            for row in records
            if all(
                row.get(column) is not None
                and row.get(column) != ""
                for column in columns
            )
        ]

    # ========================================================
    # DESCRIBE
    # ========================================================

    def describe(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return basic information about a table."""

        records = [
            dict(row)
            for row in rows
        ]

        columns = self.columns(records)

        return {
            "rows": len(records),
            "columns": len(columns),
            "column_names": columns,
            "missing_values": {
                column: sum(
                    1
                    for row in records
                    if row.get(column) is None
                    or row.get(column) == ""
                )
                for column in columns
            },
            "unique_values": {
                column: len(
                    self.unique(
                        records,
                        column,
                    )
                )
                for column in columns
            },
        }

    # ========================================================
    # EXPORT CSV
    # ========================================================

    def to_csv(
        self,
        rows: Iterable[Mapping[str, Any]],
        path: str | Path,
        *,
        columns: Sequence[str] | None = None,
        encoding: str | None = None,
    ) -> Path:
        """Export table to CSV."""

        records = [
            dict(row)
            for row in rows
        ]

        if columns is None:
            columns = self.columns(records)

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            newline="",
            encoding=encoding or self.default_encoding,
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(columns),
                extrasaction="ignore",
            )

            writer.writeheader()

            for row in records:
                writer.writerow(row)

        return output

    # ========================================================
    # EXPORT JSON
    # ========================================================

    def to_json(
        self,
        rows: Iterable[Mapping[str, Any]],
        path: str | Path,
        *,
        indent: int = 2,
        encoding: str | None = None,
    ) -> Path:
        """Export table to JSON."""

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            encoding=encoding or self.default_encoding,
        ) as file:
            json.dump(
                [dict(row) for row in rows],
                file,
                indent=indent,
                ensure_ascii=False,
                default=self._serialize,
            )

        return output

    # ========================================================
    # TEXT TABLE
    # ========================================================

    def to_text(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        columns: Sequence[str] | None = None,
        max_width: int = 40,
    ) -> str:
        """Create a simple readable text table."""

        records = [
            dict(row)
            for row in rows
        ]

        if columns is None:
            columns = self.columns(records)

        columns = [
            str(column)
            for column in columns
        ]

        if not columns:
            return ""

        widths = {
            column: min(
                max(
                    len(column),
                    max(
                        (
                            len(
                                self._display_value(
                                    row.get(column)
                                )
                            )
                            for row in records
                        ),
                        default=0,
                    ),
                ),
                max_width,
            )
            for column in columns
        }

        def format_row(row: Mapping[str, Any]) -> str:
            values = []

            for column in columns:
                value = self._display_value(
                    row.get(column)
                )

                value = value[:max_width]

                values.append(
                    value.ljust(
                        widths[column]
                    )
                )

            return " | ".join(values)

        separator = "-+-".join(
            "-" * widths[column]
            for column in columns
        )

        lines = [
            format_row(
                {
                    column: column
                    for column in columns
                }
            ),
            separator,
        ]

        lines.extend(
            format_row(row)
            for row in records
        )

        return "\n".join(lines)

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _safe_sort_value(value: Any) -> Any:
        if value is None:
            return ""

        if isinstance(
            value,
            (int, float, str, bool),
        ):
            return value

        return str(value)

    @staticmethod
    def _hashable(value: Any) -> Any:
        try:
            hash(value)
            return value
        except TypeError:
            return json.dumps(
                value,
                sort_keys=True,
                default=str,
            )

    @staticmethod
    def _aggregate_values(
        values: Iterable[Any],
        operation: str,
    ) -> Any:
        numbers = []

        for value in values:
            try:
                if value is None or isinstance(
                    value,
                    bool,
                ):
                    continue

                numbers.append(float(value))

            except (TypeError, ValueError):
                continue

        if operation == "count":
            return len(
                [
                    value
                    for value in values
                    if value is not None
                ]
            )

        if not numbers:
            return None

        if operation == "sum":
            return sum(numbers)

        if operation == "mean":
            return sum(numbers) / len(numbers)

        if operation == "min":
            return min(numbers)

        if operation == "max":
            return max(numbers)

        raise ValueError(
            f"Unsupported aggregation: {operation}"
        )

    @staticmethod
    def _merge_rows(
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(left)

        for key, value in right.items():
            if key not in result:
                result[key] = value
            else:
                result[f"right_{key}"] = value

        return result

    @staticmethod
    def _display_value(
        value: Any,
    ) -> str:
        if value is None:
            return ""

        if isinstance(value, float):
            return f"{value:g}"

        return str(value)

    @staticmethod
    def _serialize(
        value: Any,
    ) -> Any:
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
            f"TableEngine("
            f"encoding={self.default_encoding!r})"
        )


