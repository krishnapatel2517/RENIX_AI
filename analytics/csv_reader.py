"""
RENIX Analytics - CSV Reader
============================

CSV loading and writing utilities used by the analytics subsystem.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping


class CSVReader:
    """Read, inspect, and write CSV datasets."""

    def __init__(
        self,
        encoding: str = "utf-8-sig",
        delimiter: str = ",",
        quotechar: str = '"',
    ) -> None:
        self.encoding = encoding
        self.delimiter = delimiter
        self.quotechar = quotechar

    # ========================================================
    # READ
    # ========================================================

    def read(
        self,
        path: str | Path,
        *,
        delimiter: str | None = None,
        encoding: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read a CSV file into a list of dictionaries."""

        file_path = self._validate_path(path)

        with file_path.open(
            "r",
            encoding=encoding or self.encoding,
            newline="",
        ) as file:
            reader = csv.DictReader(
                file,
                delimiter=delimiter or self.delimiter,
                quotechar=self.quotechar,
            )

            if reader.fieldnames is None:
                return []

            return [
                self._clean_row(row)
                for row in reader
            ]

    # ========================================================
    # READ AS ROWS
    # ========================================================

    def read_rows(
        self,
        path: str | Path,
        *,
        delimiter: str | None = None,
        encoding: str | None = None,
    ) -> list[list[str]]:
        """Read CSV as raw rows."""

        file_path = self._validate_path(path)

        with file_path.open(
            "r",
            encoding=encoding or self.encoding,
            newline="",
        ) as file:
            reader = csv.reader(
                file,
                delimiter=delimiter or self.delimiter,
                quotechar=self.quotechar,
            )

            return [list(row) for row in reader]

    # ========================================================
    # HEADERS
    # ========================================================

    def headers(
        self,
        path: str | Path,
        *,
        delimiter: str | None = None,
        encoding: str | None = None,
    ) -> list[str]:
        """Return CSV column names."""

        file_path = self._validate_path(path)

        with file_path.open(
            "r",
            encoding=encoding or self.encoding,
            newline="",
        ) as file:
            reader = csv.reader(
                file,
                delimiter=delimiter or self.delimiter,
                quotechar=self.quotechar,
            )

            try:
                return next(reader)
            except StopIteration:
                return []

    # ========================================================
    # SAMPLE
    # ========================================================

    def sample(
        self,
        path: str | Path,
        rows: int = 5,
        *,
        delimiter: str | None = None,
        encoding: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read only the first N records."""

        if rows < 0:
            raise ValueError("rows must be >= 0")

        file_path = self._validate_path(path)

        result: list[dict[str, Any]] = []

        with file_path.open(
            "r",
            encoding=encoding or self.encoding,
            newline="",
        ) as file:
            reader = csv.DictReader(
                file,
                delimiter=delimiter or self.delimiter,
                quotechar=self.quotechar,
            )

            for index, row in enumerate(reader):
                if index >= rows:
                    break

                result.append(self._clean_row(row))

        return result

    # ========================================================
    # COUNT
    # ========================================================

    def row_count(
        self,
        path: str | Path,
        *,
        delimiter: str | None = None,
        encoding: str | None = None,
    ) -> int:
        """Return the number of data rows, excluding headers."""

        file_path = self._validate_path(path)

        count = 0

        with file_path.open(
            "r",
            encoding=encoding or self.encoding,
            newline="",
        ) as file:
            reader = csv.reader(
                file,
                delimiter=delimiter or self.delimiter,
                quotechar=self.quotechar,
            )

            try:
                next(reader)
            except StopIteration:
                return 0

            for _ in reader:
                count += 1

        return count

    # ========================================================
    # WRITE
    # ========================================================

    def write(
        self,
        path: str | Path,
        rows: Iterable[Mapping[str, Any]],
        *,
        fieldnames: Iterable[str] | None = None,
        delimiter: str | None = None,
        encoding: str | None = None,
    ) -> Path:
        """Write dictionaries to a CSV file."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        rows_list = [
            dict(row)
            for row in rows
        ]

        if fieldnames is None:
            columns = self._infer_fieldnames(rows_list)
        else:
            columns = [
                str(field)
                for field in fieldnames
            ]

        with output.open(
            "w",
            encoding=encoding or self.encoding,
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=columns,
                delimiter=delimiter or self.delimiter,
                quotechar=self.quotechar,
                extrasaction="ignore",
            )

            writer.writeheader()

            for row in rows_list:
                writer.writerow(
                    {
                        column: row.get(column, "")
                        for column in columns
                    }
                )

        return output

    # ========================================================
    # APPEND
    # ========================================================

    def append(
        self,
        path: str | Path,
        rows: Iterable[Mapping[str, Any]],
        *,
        delimiter: str | None = None,
        encoding: str | None = None,
    ) -> Path:
        """Append records to an existing CSV file."""

        file_path = self._validate_path(path)

        existing_headers = self.headers(
            file_path,
            delimiter=delimiter,
            encoding=encoding,
        )

        rows_list = [
            dict(row)
            for row in rows
        ]

        if not existing_headers:
            return self.write(
                file_path,
                rows_list,
                delimiter=delimiter,
                encoding=encoding,
            )

        with file_path.open(
            "a",
            encoding=encoding or self.encoding,
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=existing_headers,
                delimiter=delimiter or self.delimiter,
                quotechar=self.quotechar,
                extrasaction="ignore",
            )

            for row in rows_list:
                writer.writerow(
                    {
                        column: row.get(column, "")
                        for column in existing_headers
                    }
                )

        return file_path

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(
        self,
        path: str | Path,
        *,
        delimiter: str | None = None,
        encoding: str | None = None,
    ) -> dict[str, Any]:
        """Validate a CSV file and return basic structural information."""

        file_path = self._validate_path(path)

        errors: list[str] = []
        header: list[str] = []
        rows = 0

        try:
            with file_path.open(
                "r",
                encoding=encoding or self.encoding,
                newline="",
            ) as file:
                reader = csv.reader(
                    file,
                    delimiter=delimiter or self.delimiter,
                    quotechar=self.quotechar,
                )

                try:
                    header = next(reader)
                except StopIteration:
                    return {
                        "valid": True,
                        "empty": True,
                        "headers": [],
                        "rows": 0,
                        "errors": [],
                    }

                if not header:
                    errors.append("CSV contains no header columns.")

                if len(header) != len(set(header)):
                    errors.append("Duplicate column names detected.")

                expected_columns = len(header)

                for line_number, row in enumerate(
                    reader,
                    start=2,
                ):
                    rows += 1

                    if len(row) != expected_columns:
                        errors.append(
                            f"Line {line_number}: expected "
                            f"{expected_columns} columns, "
                            f"found {len(row)}."
                        )

        except UnicodeDecodeError as exc:
            errors.append(f"Encoding error: {exc}")

        except csv.Error as exc:
            errors.append(f"CSV parsing error: {exc}")

        return {
            "valid": not errors,
            "empty": rows == 0 and not header,
            "headers": header,
            "rows": rows,
            "errors": errors,
        }

    # ========================================================
    # TYPE INFERENCE
    # ========================================================

    def infer_types(
        self,
        path: str | Path,
        *,
        delimiter: str | None = None,
        encoding: str | None = None,
        sample_size: int = 100,
    ) -> dict[str, str]:
        """Infer basic column types from a CSV sample."""

        rows = self.sample(
            path,
            rows=sample_size,
            delimiter=delimiter,
            encoding=encoding,
        )

        if not rows:
            return {}

        result: dict[str, str] = {}

        columns = list(rows[0].keys())

        for column in columns:
            values = [
                row.get(column)
                for row in rows
                if row.get(column) not in (None, "")
            ]

            result[column] = self._infer_column_type(values)

        return result

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _validate_path(path: str | Path) -> Path:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"CSV path is not a file: {file_path}"
            )

        return file_path

    @staticmethod
    def _clean_row(
        row: Mapping[str | None, str | None],
    ) -> dict[str, Any]:
        """Clean a DictReader row."""

        cleaned: dict[str, Any] = {}

        for key, value in row.items():
            if key is None:
                continue

            normalized_key = str(key).strip()

            if isinstance(value, str):
                value = value.strip()

            cleaned[normalized_key] = value

        return cleaned

    @staticmethod
    def _infer_fieldnames(
        rows: list[Mapping[str, Any]],
    ) -> list[str]:
        columns: list[str] = []

        for row in rows:
            for key in row.keys():
                key = str(key)

                if key not in columns:
                    columns.append(key)

        return columns

    @staticmethod
    def _infer_column_type(values: list[Any]) -> str:
        if not values:
            return "unknown"

        if all(
            isinstance(value, bool)
            or str(value).strip().lower() in {"true", "false"}
            for value in values
        ):
            return "boolean"

        try:
            for value in values:
                int(str(value).strip())

            return "integer"
        except (ValueError, TypeError):
            pass

        try:
            for value in values:
                float(str(value).strip())

            return "float"
        except (ValueError, TypeError):
            pass

        return "string"

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            f"CSVReader("
            f"encoding={self.encoding!r}, "
            f"delimiter={self.delimiter!r})"
        )



