"""
RENIX Analytics - Excel Reader
==============================

Utilities for reading and writing Excel workbooks.

The reader keeps Excel support optional so RENIX can still start when
an Excel dependency is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


class ExcelReader:
    """Read, inspect, and write Excel workbooks."""

    SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}

    def __init__(self) -> None:
        self._openpyxl = None

    # ========================================================
    # DEPENDENCY
    # ========================================================

    def _get_openpyxl(self):
        """Load openpyxl lazily."""

        if self._openpyxl is None:
            try:
                import openpyxl
            except ImportError as exc:
                raise RuntimeError(
                    "Excel support requires 'openpyxl'. "
                    "Install it with: pip install openpyxl"
                ) from exc

            self._openpyxl = openpyxl

        return self._openpyxl

    # ========================================================
    # READ
    # ========================================================

    def read(
        self,
        path: str | Path,
        *,
        sheet: str | int | None = None,
        data_only: bool = True,
        header: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Read an Excel sheet into a list of dictionaries.

        If `sheet` is omitted, the active sheet is used.
        """

        workbook = self._load_workbook(
            path,
            data_only=data_only,
        )

        worksheet = self._get_sheet(workbook, sheet)

        rows = list(
            worksheet.iter_rows(
                values_only=True
            )
        )

        if not rows:
            return []

        if header:
            headers = self._build_headers(rows[0])
            data_rows = rows[1:]
        else:
            headers = [
                f"column_{index + 1}"
                for index in range(
                    max(len(row) for row in rows)
                )
            ]
            data_rows = rows

        return [
            self._row_to_dict(row, headers)
            for row in data_rows
            if any(value is not None for value in row)
        ]

    # ========================================================
    # RAW ROWS
    # ========================================================

    def read_rows(
        self,
        path: str | Path,
        *,
        sheet: str | int | None = None,
        data_only: bool = True,
    ) -> list[list[Any]]:
        """Read a worksheet as raw rows."""

        workbook = self._load_workbook(
            path,
            data_only=data_only,
        )

        worksheet = self._get_sheet(workbook, sheet)

        return [
            list(row)
            for row in worksheet.iter_rows(
                values_only=True
            )
        ]

    # ========================================================
    # SHEETS
    # ========================================================

    def sheets(
        self,
        path: str | Path,
    ) -> list[str]:
        """Return all worksheet names."""

        workbook = self._load_workbook(path)

        return list(workbook.sheetnames)

    def sheet_count(
        self,
        path: str | Path,
    ) -> int:
        """Return the number of worksheets."""

        return len(self.sheets(path))

    def active_sheet(
        self,
        path: str | Path,
    ) -> str:
        """Return the active worksheet name."""

        workbook = self._load_workbook(path)

        return workbook.active.title

    # ========================================================
    # HEADERS
    # ========================================================

    def headers(
        self,
        path: str | Path,
        *,
        sheet: str | int | None = None,
    ) -> list[str]:
        """Return the first row as column names."""

        rows = self.read_rows(path, sheet=sheet)

        if not rows:
            return []

        return self._build_headers(rows[0])

    # ========================================================
    # SAMPLE
    # ========================================================

    def sample(
        self,
        path: str | Path,
        rows: int = 5,
        *,
        sheet: str | int | None = None,
        data_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Read only the first N data rows."""

        if rows < 0:
            raise ValueError("rows must be >= 0")

        workbook = self._load_workbook(
            path,
            data_only=data_only,
        )

        worksheet = self._get_sheet(workbook, sheet)

        iterator = worksheet.iter_rows(
            values_only=True
        )

        try:
            header_row = next(iterator)
        except StopIteration:
            return []

        headers = self._build_headers(header_row)

        result = []

        for index, row in enumerate(iterator):
            if index >= rows:
                break

            if any(value is not None for value in row):
                result.append(
                    self._row_to_dict(row, headers)
                )

        return result

    # ========================================================
    # CELL ACCESS
    # ========================================================

    def get_cell(
        self,
        path: str | Path,
        cell: str,
        *,
        sheet: str | int | None = None,
        data_only: bool = True,
    ) -> Any:
        """Read a single cell."""

        workbook = self._load_workbook(
            path,
            data_only=data_only,
        )

        worksheet = self._get_sheet(workbook, sheet)

        return worksheet[cell].value

    # ========================================================
    # WRITE
    # ========================================================

    def write(
        self,
        path: str | Path,
        rows: Iterable[Mapping[str, Any]],
        *,
        sheet: str = "Sheet1",
        overwrite: bool = True,
    ) -> Path:
        """Write record-oriented data to an Excel workbook."""

        openpyxl = self._get_openpyxl()

        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {output}"
            )

        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        worksheet.title = sheet

        rows_list = [
            dict(row)
            for row in rows
        ]

        columns = self._infer_columns(rows_list)

        if columns:
            worksheet.append(columns)

            for row in rows_list:
                worksheet.append(
                    [
                        row.get(column)
                        for column in columns
                    ]
                )

        workbook.save(output)

        return output

    # ========================================================
    # WRITE MULTIPLE SHEETS
    # ========================================================

    def write_workbook(
        self,
        path: str | Path,
        sheets: Mapping[
            str,
            Iterable[Mapping[str, Any]]
        ],
        *,
        overwrite: bool = True,
    ) -> Path:
        """Write multiple datasets into one workbook."""

        openpyxl = self._get_openpyxl()

        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {output}"
            )

        workbook = openpyxl.Workbook()

        # Remove the automatically created sheet.
        default_sheet = workbook.active
        workbook.remove(default_sheet)

        for sheet_name, rows in sheets.items():
            worksheet = workbook.create_sheet(
                title=self._safe_sheet_name(sheet_name)
            )

            rows_list = [
                dict(row)
                for row in rows
            ]

            columns = self._infer_columns(
                rows_list
            )

            if columns:
                worksheet.append(columns)

                for row in rows_list:
                    worksheet.append(
                        [
                            row.get(column)
                            for column in columns
                        ]
                    )

        if not workbook.sheetnames:
            workbook.create_sheet("Sheet1")

        workbook.save(output)

        return output

    # ========================================================
    # APPEND
    # ========================================================

    def append(
        self,
        path: str | Path,
        rows: Iterable[Mapping[str, Any]],
        *,
        sheet: str | int | None = None,
    ) -> Path:
        """Append records to an existing worksheet."""

        openpyxl = self._get_openpyxl()

        file_path = self._validate_path(path)

        workbook = openpyxl.load_workbook(
            file_path
        )

        worksheet = self._get_sheet(
            workbook,
            sheet,
        )

        rows_list = [
            dict(row)
            for row in rows
        ]

        if not rows_list:
            workbook.save(file_path)
            return file_path

        existing_headers = [
            cell.value
            for cell in worksheet[1]
        ]

        existing_headers = [
            str(header)
            for header in existing_headers
            if header is not None
        ]

        if not existing_headers:
            existing_headers = self._infer_columns(
                rows_list
            )

            for index, column in enumerate(
                existing_headers,
                start=1,
            ):
                worksheet.cell(
                    row=1,
                    column=index,
                    value=column,
                )

        for row in rows_list:
            worksheet.append(
                [
                    row.get(column)
                    for column in existing_headers
                ]
            )

        workbook.save(file_path)

        return file_path

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        """Validate an Excel workbook."""

        try:
            workbook = self._load_workbook(path)

            information = {
                "valid": True,
                "sheets": list(workbook.sheetnames),
                "sheet_count": len(
                    workbook.sheetnames
                ),
                "errors": [],
            }

            for worksheet in workbook.worksheets:
                information.setdefault(
                    "dimensions",
                    {},
                )[worksheet.title] = {
                    "rows": worksheet.max_row,
                    "columns": worksheet.max_column,
                }

            return information

        except Exception as exc:
            return {
                "valid": False,
                "sheets": [],
                "sheet_count": 0,
                "errors": [str(exc)],
            }

    # ========================================================
    # DIMENSIONS
    # ========================================================

    def dimensions(
        self,
        path: str | Path,
        *,
        sheet: str | int | None = None,
    ) -> tuple[int, int]:
        """Return worksheet dimensions as (rows, columns)."""

        workbook = self._load_workbook(path)

        worksheet = self._get_sheet(
            workbook,
            sheet,
        )

        return (
            worksheet.max_row,
            worksheet.max_column,
        )

    # ========================================================
    # TYPE INFERENCE
    # ========================================================

    def infer_types(
        self,
        path: str | Path,
        *,
        sheet: str | int | None = None,
        sample_size: int = 100,
    ) -> dict[str, str]:
        """Infer basic column types."""

        rows = self.sample(
            path,
            rows=sample_size,
            sheet=sheet,
        )

        if not rows:
            return {}

        columns = list(rows[0].keys())
        result: dict[str, str] = {}

        for column in columns:
            values = [
                row.get(column)
                for row in rows
                if row.get(column) is not None
            ]

            result[column] = self._infer_type(values)

        return result

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    def _load_workbook(
        self,
        path: str | Path,
        *,
        data_only: bool = True,
    ):
        openpyxl = self._get_openpyxl()

        file_path = self._validate_path(path)

        return openpyxl.load_workbook(
            file_path,
            data_only=data_only,
            read_only=False,
        )

    def _get_sheet(
        self,
        workbook,
        sheet: str | int | None,
    ):
        if sheet is None:
            return workbook.active

        if isinstance(sheet, int):
            if sheet < 0 or sheet >= len(
                workbook.worksheets
            ):
                raise IndexError(
                    f"Worksheet index out of range: {sheet}"
                )

            return workbook.worksheets[sheet]

        if sheet not in workbook.sheetnames:
            raise KeyError(
                f"Worksheet not found: {sheet}"
            )

        return workbook[sheet]

    @staticmethod
    def _validate_path(
        path: str | Path,
    ) -> Path:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Excel file not found: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Excel path is not a file: {file_path}"
            )

        if file_path.suffix.lower() not in {
            ".xlsx",
            ".xlsm",
            ".xltx",
            ".xltm",
            ".xls",
        }:
            raise ValueError(
                f"Unsupported Excel extension: "
                f"{file_path.suffix}"
            )

        return file_path

    @staticmethod
    def _build_headers(
        row: Iterable[Any],
    ) -> list[str]:
        headers: list[str] = []
        used: dict[str, int] = {}

        for index, value in enumerate(row):
            name = (
                str(value).strip()
                if value is not None
                else ""
            )

            if not name:
                name = f"column_{index + 1}"

            base_name = name

            if name in used:
                used[name] += 1
                name = (
                    f"{base_name}_{used[base_name]}"
                )
            else:
                used[name] = 0

            headers.append(name)

        return headers

    @staticmethod
    def _row_to_dict(
        row: Iterable[Any],
        headers: list[str],
    ) -> dict[str, Any]:
        values = list(row)

        return {
            header: (
                values[index]
                if index < len(values)
                else None
            )
            for index, header in enumerate(headers)
        }

    @staticmethod
    def _infer_columns(
        rows: list[Mapping[str, Any]],
    ) -> list[str]:
        columns: list[str] = []

        for row in rows:
            for key in row:
                key = str(key)

                if key not in columns:
                    columns.append(key)

        return columns

    @staticmethod
    def _infer_type(
        values: list[Any],
    ) -> str:
        if not values:
            return "unknown"

        if all(
            isinstance(value, bool)
            for value in values
        ):
            return "boolean"

        if all(
            isinstance(value, int)
            and not isinstance(value, bool)
            for value in values
        ):
            return "integer"

        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            for value in values
        ):
            return "number"

        if all(
            hasattr(value, "year")
            and hasattr(value, "month")
            and hasattr(value, "day")
            for value in values
        ):
            return "date"

        return "string"

    @staticmethod
    def _safe_sheet_name(name: str) -> str:
        invalid = {
            ":",
            "\\",
            "/",
            "?",
            "*",
            "[",
            "]",
        }

        cleaned = "".join(
            "_" if char in invalid else char
            for char in str(name)
        )

        cleaned = cleaned.strip() or "Sheet"

        return cleaned[:31]

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return "ExcelReader()"



