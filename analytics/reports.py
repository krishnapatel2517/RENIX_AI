"""
RENIX Analytics - Reports
=========================

Report generation utilities for RENIX analytics.

Supports:
- Dataset summaries
- Statistical reports
- Table reports
- JSON/HTML/Markdown export
- Automatic report metadata
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ReportEngine:
    """Generate structured analytics reports."""

    def __init__(
        self,
        output_directory: str | Path = "data/statistics/reports",
    ) -> None:
        self.output_directory = Path(output_directory)

    # ========================================================
    # REPORT CREATION
    # ========================================================

    def create_report(
        self,
        title: str,
        *,
        description: str = "",
        data: Any = None,
        statistics: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a standard RENIX analytics report."""

        return {
            "report": {
                "title": title,
                "description": description,
                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "generator": "RENIX Analytics",
            },
            "metadata": dict(metadata or {}),
            "statistics": dict(statistics or {}),
            "data": data,
        }

    # ========================================================
    # DATASET REPORT
    # ========================================================

    def dataset_report(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        title: str = "Dataset Analytics Report",
        description: str = "",
    ) -> dict[str, Any]:
        """Generate a report describing a dataset."""

        records = [
            dict(row)
            for row in rows
        ]

        columns = self._columns(records)

        numeric_summary: dict[str, Any] = {}

        for column in columns:
            values = [
                row.get(column)
                for row in records
            ]

            numbers = self._numbers(values)

            if numbers:
                numeric_summary[column] = (
                    self._describe_numbers(numbers)
                )

        missing = {
            column: sum(
                1
                for row in records
                if row.get(column) is None
                or row.get(column) == ""
            )
            for column in columns
        }

        return self.create_report(
            title,
            description=description,
            data={
                "rows": records,
                "row_count": len(records),
                "columns": columns,
            },
            statistics={
                "row_count": len(records),
                "column_count": len(columns),
                "missing_values": missing,
                "numeric_columns": numeric_summary,
            },
        )

    # ========================================================
    # COMPARISON REPORT
    # ========================================================

    def comparison_report(
        self,
        datasets: Mapping[str, Iterable[Mapping[str, Any]]],
        *,
        title: str = "Dataset Comparison Report",
    ) -> dict[str, Any]:
        """Compare multiple datasets."""

        summaries: dict[str, Any] = {}

        for name, rows in datasets.items():
            records = [
                dict(row)
                for row in rows
            ]

            summaries[str(name)] = {
                "row_count": len(records),
                "columns": self._columns(records),
                "numeric": self._numeric_columns(
                    records
                ),
            }

        return self.create_report(
            title,
            data=summaries,
        )

    # ========================================================
    # STATISTICS REPORT
    # ========================================================

    def statistics_report(
        self,
        statistics: Mapping[str, Any],
        *,
        title: str = "Statistical Analysis Report",
        description: str = "",
    ) -> dict[str, Any]:
        """Create a report from precomputed statistics."""

        return self.create_report(
            title,
            description=description,
            statistics=statistics,
        )

    # ========================================================
    # TABLE REPORT
    # ========================================================

    def table_report(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        title: str = "Analytics Table Report",
        columns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Create a report focused on tabular data."""

        records = [
            dict(row)
            for row in rows
        ]

        if columns is None:
            columns = self._columns(records)

        return self.create_report(
            title,
            data={
                "columns": list(columns),
                "rows": records,
            },
            statistics={
                "row_count": len(records),
                "column_count": len(columns),
            },
        )

    # ========================================================
    # JSON EXPORT
    # ========================================================

    def save_json(
        self,
        report: Mapping[str, Any],
        path: str | Path | None = None,
        *,
        indent: int = 2,
    ) -> Path:
        """Save a report as JSON."""

        output = self._prepare_output(
            path,
            "analytics_report.json",
        )

        with output.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=indent,
                ensure_ascii=False,
                default=self._serialize,
            )

        return output

    # ========================================================
    # MARKDOWN EXPORT
    # ========================================================

    def save_markdown(
        self,
        report: Mapping[str, Any],
        path: str | Path | None = None,
    ) -> Path:
        """Save a report as Markdown."""

        output = self._prepare_output(
            path,
            "analytics_report.md",
        )

        markdown = self.to_markdown(report)

        output.write_text(
            markdown,
            encoding="utf-8",
        )

        return output

    def to_markdown(
        self,
        report: Mapping[str, Any],
    ) -> str:
        """Convert a report to Markdown."""

        report_info = report.get(
            "report",
            {},
        )

        title = report_info.get(
            "title",
            "RENIX Analytics Report",
        )

        description = report_info.get(
            "description",
            "",
        )

        generated_at = report_info.get(
            "generated_at",
            "",
        )

        lines = [
            f"# {title}",
            "",
        ]

        if description:
            lines.extend(
                [
                    description,
                    "",
                ]
            )

        if generated_at:
            lines.extend(
                [
                    f"**Generated:** {generated_at}",
                    "",
                ]
            )

        statistics = report.get(
            "statistics",
            {},
        )

        if statistics:
            lines.extend(
                [
                    "## Statistics",
                    "",
                ]
            )

            lines.extend(
                self._markdown_mapping(
                    statistics
                )
            )

            lines.append("")

        data = report.get(
            "data"
        )

        if data is not None:
            lines.extend(
                [
                    "## Data",
                    "",
                ]
            )

            if self._is_table_data(data):
                lines.append(
                    self._markdown_table(
                        data
                    )
                )
            else:
                lines.append(
                    "```json"
                )
                lines.append(
                    json.dumps(
                        data,
                        indent=2,
                        ensure_ascii=False,
                        default=self._serialize,
                    )
                )
                lines.append(
                    "```"
                )

            lines.append("")

        metadata = report.get(
            "metadata",
            {},
        )

        if metadata:
            lines.extend(
                [
                    "## Metadata",
                    "",
                ]
            )

            lines.extend(
                self._markdown_mapping(
                    metadata
                )
            )

        return "\n".join(lines).rstrip() + "\n"

    # ========================================================
    # HTML EXPORT
    # ========================================================

    def save_html(
        self,
        report: Mapping[str, Any],
        path: str | Path | None = None,
    ) -> Path:
        """Save a report as HTML."""

        output = self._prepare_output(
            path,
            "analytics_report.html",
        )

        html = self.to_html(report)

        output.write_text(
            html,
            encoding="utf-8",
        )

        return output

    def to_html(
        self,
        report: Mapping[str, Any],
    ) -> str:
        """Convert a report to standalone HTML."""

        report_info = report.get(
            "report",
            {},
        )

        title = self._escape(
            str(
                report_info.get(
                    "title",
                    "RENIX Analytics Report",
                )
            )
        )

        description = self._escape(
            str(
                report_info.get(
                    "description",
                    "",
                )
            )
        )

        generated_at = self._escape(
            str(
                report_info.get(
                    "generated_at",
                    "",
                )
            )
        )

        html: list[str] = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{title}</title>",
            "<style>",
            "body {",
            "  font-family: Arial, sans-serif;",
            "  margin: 40px;",
            "  line-height: 1.5;",
            "}",
            "table {",
            "  border-collapse: collapse;",
            "  width: 100%;",
            "  margin: 20px 0;",
            "}",
            "th, td {",
            "  border: 1px solid #ccc;",
            "  padding: 8px;",
            "  text-align: left;",
            "}",
            "th {",
            "  font-weight: bold;",
            "}",
            ".metric {",
            "  margin: 8px 0;",
            "}",
            "pre {",
            "  background: #f5f5f5;",
            "  padding: 15px;",
            "  overflow-x: auto;",
            "}",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
        ]

        if description:
            html.append(
                f"<p>{description}</p>"
            )

        if generated_at:
            html.append(
                f"<p><strong>Generated:</strong> "
                f"{generated_at}</p>"
            )

        statistics = report.get(
            "statistics",
            {},
        )

        if statistics:
            html.append(
                "<h2>Statistics</h2>"
            )
            html.append(
                self._html_mapping(
                    statistics
                )
            )

        data = report.get(
            "data"
        )

        if data is not None:
            html.append(
                "<h2>Data</h2>"
            )

            if self._is_table_data(data):
                html.append(
                    self._html_table(
                        data
                    )
                )
            else:
                encoded = self._escape(
                    json.dumps(
                        data,
                        indent=2,
                        ensure_ascii=False,
                        default=self._serialize,
                    )
                )

                html.append(
                    f"<pre>{encoded}</pre>"
                )

        metadata = report.get(
            "metadata",
            {},
        )

        if metadata:
            html.append(
                "<h2>Metadata</h2>"
            )
            html.append(
                self._html_mapping(
                    metadata
                )
            )

        html.extend(
            [
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(html)

    # ========================================================
    # SUMMARY TEXT
    # ========================================================

    def summary(
        self,
        report: Mapping[str, Any],
    ) -> str:
        """Create a short human-readable report summary."""

        info = report.get(
            "report",
            {},
        )

        title = info.get(
            "title",
            "RENIX Analytics Report",
        )

        statistics = report.get(
            "statistics",
            {},
        )

        lines = [
            str(title),
            "=" * len(str(title)),
        ]

        for key, value in statistics.items():
            if isinstance(value, Mapping):
                lines.append(
                    f"{key}:"
                )

                for sub_key, sub_value in value.items():
                    lines.append(
                        f"  {sub_key}: "
                        f"{self._format_value(sub_value)}"
                    )
            else:
                lines.append(
                    f"{key}: "
                    f"{self._format_value(value)}"
                )

        return "\n".join(lines)

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    def _prepare_output(
        self,
        path: str | Path | None,
        default_name: str,
    ) -> Path:
        output = (
            self.output_directory / default_name
            if path is None
            else Path(path)
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output

    @staticmethod
    def _columns(
        rows: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        columns: list[str] = []

        for row in rows:
            for key in row:
                key = str(key)

                if key not in columns:
                    columns.append(key)

        return columns

    def _numeric_columns(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}

        for column in self._columns(rows):
            numbers = self._numbers(
                row.get(column)
                for row in rows
            )

            if numbers:
                result[column] = (
                    self._describe_numbers(
                        numbers
                    )
                )

        return result

    @staticmethod
    def _numbers(
        values: Iterable[Any],
    ) -> list[float]:
        numbers: list[float] = []

        for value in values:
            if value is None or value == "":
                continue

            if isinstance(value, bool):
                continue

            try:
                number = float(value)

                if number == number:
                    numbers.append(number)

            except (TypeError, ValueError):
                continue

        return numbers

    @staticmethod
    def _describe_numbers(
        numbers: Sequence[float],
    ) -> dict[str, float | int | None]:
        if not numbers:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": None,
                "minimum": None,
                "maximum": None,
            }

        total = sum(numbers)

        return {
            "count": len(numbers),
            "sum": total,
            "mean": total / len(numbers),
            "minimum": min(numbers),
            "maximum": max(numbers),
        }

    @staticmethod
    def _is_table_data(
        data: Any,
    ) -> bool:
        return (
            isinstance(data, Mapping)
            and isinstance(
                data.get("columns"),
                (list, tuple),
            )
            and isinstance(
                data.get("rows"),
                list,
            )
        )

    def _markdown_table(
        self,
        data: Mapping[str, Any],
    ) -> str:
        columns = [
            str(column)
            for column in data["columns"]
        ]

        rows = data["rows"]

        header = (
            "| "
            + " | ".join(columns)
            + " |"
        )

        separator = (
            "| "
            + " | ".join(
                "---"
                for _ in columns
            )
            + " |"
        )

        lines = [
            header,
            separator,
        ]

        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    self._markdown_value(
                        row.get(column)
                    )
                    for column in columns
                )
                + " |"
            )

        return "\n".join(lines)

    def _html_table(
        self,
        data: Mapping[str, Any],
    ) -> str:
        columns = [
            str(column)
            for column in data["columns"]
        ]

        rows = data["rows"]

        html = [
            "<table>",
            "<thead>",
            "<tr>",
        ]

        for column in columns:
            html.append(
                f"<th>{self._escape(column)}</th>"
            )

        html.extend(
            [
                "</tr>",
                "</thead>",
                "<tbody>",
            ]
        )

        for row in rows:
            html.append("<tr>")

            for column in columns:
                html.append(
                    "<td>"
                    + self._escape(
                        self._format_value(
                            row.get(column)
                        )
                    )
                    + "</td>"
                )

            html.append("</tr>")

        html.extend(
            [
                "</tbody>",
                "</table>",
            ]
        )

        return "\n".join(html)

    def _markdown_mapping(
        self,
        mapping: Mapping[str, Any],
        level: int = 0,
    ) -> list[str]:
        lines: list[str] = []

        for key, value in mapping.items():
            if isinstance(value, Mapping):
                lines.append(
                    f"- **{key}**:"
                )

                lines.extend(
                    self._markdown_mapping(
                        value,
                        level + 1,
                    )
                )
            else:
                lines.append(
                    f"- **{key}:** "
                    f"{self._markdown_value(value)}"
                )

        return lines

    def _html_mapping(
        self,
        mapping: Mapping[str, Any],
    ) -> str:
        items: list[str] = [
            "<div>"
        ]

        for key, value in mapping.items():
            if isinstance(value, Mapping):
                items.append(
                    f"<div class='metric'>"
                    f"<strong>"
                    f"{self._escape(str(key))}"
                    f"</strong></div>"
                )

                items.append(
                    self._html_mapping(value)
                )
            else:
                items.append(
                    "<div class='metric'>"
                    f"<strong>{self._escape(str(key))}"
                    f":</strong> "
                    f"{self._escape(self._format_value(value))}"
                    "</div>"
                )

        items.append("</div>")

        return "\n".join(items)

    @staticmethod
    def _markdown_value(
        value: Any,
    ) -> str:
        text = ReportEngine._format_value(value)

        return (
            text
            .replace("|", "\\|")
            .replace("\n", " ")
        )

    @staticmethod
    def _format_value(
        value: Any,
    ) -> str:
        if value is None:
            return "N/A"

        if isinstance(value, float):
            return f"{value:.4f}"

        if isinstance(
            value,
            (dict, list, tuple),
        ):
            return json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )

        return str(value)

    @staticmethod
    def _escape(
        value: str,
    ) -> str:
        return (
            value
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

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
            f"ReportEngine("
            f"output_directory="
            f"{str(self.output_directory)!r})"
        )



