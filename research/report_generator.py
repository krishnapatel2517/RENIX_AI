"""
RENIX Research Report Generator

Generates structured research reports from:
- Research sources
- Summaries
- Comparisons
- Findings
- Claims
- Citations

Designed to work with the other modules inside:
    RENIX/research/

The generator itself does not perform web searching.
It transforms research data into a clean report structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable
import re


# ================================================================
# DATA MODELS
# ================================================================


@dataclass
class ReportSection:
    """Represents one section of a research report."""

    title: str
    content: str = ""
    level: int = 1
    bullets: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "bullets": list(self.bullets),
        }


@dataclass
class ResearchReport:
    """Complete generated research report."""

    title: str
    topic: str
    created_at: str
    sections: list[ReportSection] = field(
        default_factory=list
    )
    citations: list[str] = field(
        default_factory=list
    )
    sources: list[Any] = field(
        default_factory=list
    )
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "topic": self.topic,
            "created_at": self.created_at,
            "sections": [
                section.to_dict()
                for section in self.sections
            ],
            "citations": list(
                self.citations
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# ================================================================
# REPORT GENERATOR
# ================================================================


class ReportGenerator:
    """
    Generates professional research reports for RENIX.

    Supported output formats:
        - markdown
        - plain text
        - html
        - structured dictionary

    The class is intentionally independent of the browser
    and research engines so it can also generate reports from
    locally supplied information.
    """

    def __init__(
        self,
        *,
        default_title: str = "RENIX Research Report",
    ) -> None:

        self.default_title = (
            default_title
        )

    # ============================================================
    # MAIN API
    # ============================================================

    def generate(
        self,
        topic: str,
        *,
        summary: str = "",
        findings: Iterable[str] | None = None,
        sources: Iterable[Any] | None = None,
        citations: Iterable[str] | None = None,
        comparison: Any = None,
        key_points: Iterable[str] | None = None,
        limitations: Iterable[str] | None = None,
        recommendations: Iterable[str] | None = None,
        title: str | None = None,
    ) -> ResearchReport:

        topic = self._clean_text(
            topic
        )

        report = ResearchReport(
            title=(
                title
                or self._generate_title(
                    topic
                )
            ),
            topic=topic,
            created_at=(
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            sources=list(
                sources or []
            ),
            citations=list(
                citations or []
            ),
        )

        # --------------------------------------------------------
        # Executive Summary
        # --------------------------------------------------------

        if summary:

            report.sections.append(
                ReportSection(
                    title="Executive Summary",
                    content=self._clean_text(
                        summary
                    ),
                    level=1,
                )
            )

        # --------------------------------------------------------
        # Key Findings
        # --------------------------------------------------------

        finding_list = list(
            findings or []
        )

        if not finding_list:
            finding_list = list(
                key_points or []
            )

        if finding_list:

            report.sections.append(
                ReportSection(
                    title="Key Findings",
                    bullets=[
                        self._clean_text(
                            finding
                        )
                        for finding in finding_list
                        if self._clean_text(
                            finding
                        )
                    ],
                    level=1,
                )
            )

        # --------------------------------------------------------
        # Comparison
        # --------------------------------------------------------

        if comparison is not None:

            section = (
                self._comparison_section(
                    comparison
                )
            )

            if section:
                report.sections.append(
                    section
                )

        # --------------------------------------------------------
        # Recommendations
        # --------------------------------------------------------

        recommendation_list = [
            self._clean_text(
                item
            )
            for item in (
                recommendations or []
            )
            if self._clean_text(
                item
            )
        ]

        if recommendation_list:

            report.sections.append(
                ReportSection(
                    title="Recommendations",
                    bullets=(
                        recommendation_list
                    ),
                    level=1,
                )
            )

        # --------------------------------------------------------
        # Limitations
        # --------------------------------------------------------

        limitation_list = [
            self._clean_text(
                item
            )
            for item in (
                limitations or []
            )
            if self._clean_text(
                item
            )
        ]

        if limitation_list:

            report.sections.append(
                ReportSection(
                    title="Limitations",
                    bullets=(
                        limitation_list
                    ),
                    level=1,
                )
            )

        # --------------------------------------------------------
        # Sources
        # --------------------------------------------------------

        if report.sources:

            report.sections.append(
                self._sources_section(
                    report.sources
                )
            )

        # --------------------------------------------------------
        # References
        # --------------------------------------------------------

        if report.citations:

            report.sections.append(
                ReportSection(
                    title="References",
                    bullets=[
                        self._clean_text(
                            citation
                        )
                        for citation
                        in report.citations
                        if self._clean_text(
                            citation
                        )
                    ],
                    level=1,
                )
            )

        report.metadata = {
            "source_count": len(
                report.sources
            ),
            "citation_count": len(
                report.citations
            ),
            "section_count": len(
                report.sections
            ),
        }

        return report

    # ============================================================
    # FROM SUMMARY
    # ============================================================

    def from_summary(
        self,
        topic: str,
        summary_result: Any,
        *,
        sources: Iterable[Any] | None = None,
        citations: Iterable[str] | None = None,
        title: str | None = None,
    ) -> ResearchReport:

        summary = self._get(
            summary_result,
            "summary",
            "",
        )

        key_points = self._get(
            summary_result,
            "key_points",
            [],
        )

        extracted_points = []

        for point in (
            key_points or []
        ):

            if isinstance(
                point,
                str,
            ):
                extracted_points.append(
                    point
                )

            else:
                text = self._get(
                    point,
                    "text",
                    "",
                )

                if text:
                    extracted_points.append(
                        text
                    )

        return self.generate(
            topic,
            summary=summary,
            key_points=extracted_points,
            sources=sources,
            citations=citations,
            title=title,
        )

    # ============================================================
    # FROM COMPARISON
    # ============================================================

    def from_comparison(
        self,
        topic: str,
        comparison_result: Any,
        *,
        sources: Iterable[Any] | None = None,
        citations: Iterable[str] | None = None,
        title: str | None = None,
    ) -> ResearchReport:

        summary = self._get(
            comparison_result,
            "summary",
            "",
        )

        similarities = self._get(
            comparison_result,
            "similarities",
            [],
        )

        differences = self._get(
            comparison_result,
            "differences",
            [],
        )

        rankings = self._get(
            comparison_result,
            "rankings",
            [],
        )

        findings = []

        if summary:
            findings.append(
                summary
            )

        findings.extend(
            str(item)
            for item in similarities
            if item
        )

        findings.extend(
            str(item)
            for item in differences
            if item
        )

        if rankings:

            findings.append(
                "Ranking:"
            )

            findings.extend(
                str(item)
                for item in rankings
                if item
            )

        return self.generate(
            topic,
            summary=summary,
            findings=findings,
            comparison=comparison_result,
            sources=sources,
            citations=citations,
            title=title,
        )

    # ============================================================
    # SECTION BUILDERS
    # ============================================================

    def _comparison_section(
        self,
        comparison: Any,
    ) -> ReportSection | None:

        attributes = self._get(
            comparison,
            "attributes",
            [],
        )

        similarities = self._get(
            comparison,
            "similarities",
            [],
        )

        differences = self._get(
            comparison,
            "differences",
            [],
        )

        rankings = self._get(
            comparison,
            "rankings",
            [],
        )

        content_parts = []

        if similarities:

            content_parts.append(
                "Similarities:\n"
                + "\n".join(
                    f"- {item}"
                    for item in similarities
                )
            )

        if differences:

            content_parts.append(
                "Differences:\n"
                + "\n".join(
                    f"- {item}"
                    for item in differences
                )
            )

        if attributes:

            attribute_lines = []

            for attribute in attributes:

                name = self._get(
                    attribute,
                    "name",
                    "",
                )

                winner = self._get(
                    attribute,
                    "winner",
                    None,
                )

                explanation = self._get(
                    attribute,
                    "explanation",
                    "",
                )

                if winner:

                    line = (
                        f"{name}: "
                        f"{winner}"
                    )

                    if explanation:
                        line += (
                            f" — "
                            f"{explanation}"
                        )

                    attribute_lines.append(
                        line
                    )

            if attribute_lines:

                content_parts.append(
                    "Attribute comparison:\n"
                    + "\n".join(
                        f"- {line}"
                        for line
                        in attribute_lines
                    )
                )

        bullets = [
            str(item)
            for item in rankings
            if item
        ]

        return ReportSection(
            title="Comparison Analysis",
            content="\n\n".join(
                content_parts
            ),
            bullets=bullets,
            level=1,
        )

    def _sources_section(
        self,
        sources: list[Any],
    ) -> ReportSection:

        bullets = []

        for source in sources:

            title = self._get(
                source,
                "title",
                "",
            )

            url = self._get(
                source,
                "url",
                "",
            )

            publisher = self._get(
                source,
                "publisher",
                "",
            )

            if title:

                text = str(
                    title
                )

                if publisher:
                    text += (
                        f" — {publisher}"
                    )

                if url:
                    text += (
                        f" — {url}"
                    )

                bullets.append(
                    text
                )

            else:

                text = str(
                    source
                )

                if text:
                    bullets.append(
                        text
                    )

        return ReportSection(
            title="Sources",
            bullets=bullets,
            level=1,
        )

    # ============================================================
    # MARKDOWN
    # ============================================================

    def to_markdown(
        self,
        report: ResearchReport,
    ) -> str:

        lines = [
            f"# {report.title}",
            "",
            f"**Topic:** {report.topic}",
            "",
            f"**Generated:** "
            f"{report.created_at}",
            "",
        ]

        for section in report.sections:

            level = max(
                1,
                min(
                    6,
                    section.level,
                ),
            )

            lines.append(
                f"{'#' * (level + 1)} "
                f"{section.title}"
            )

            lines.append("")

            if section.content:

                lines.append(
                    section.content
                )

                lines.append("")

            for bullet in section.bullets:

                lines.append(
                    f"- {bullet}"
                )

            if section.bullets:
                lines.append("")

        return "\n".join(
            lines
        ).strip()

    # ============================================================
    # PLAIN TEXT
    # ============================================================

    def to_text(
        self,
        report: ResearchReport,
    ) -> str:

        lines = [
            report.title.upper(),
            "=" * len(
                report.title
            ),
            "",
            f"Topic: {report.topic}",
            f"Generated: "
            f"{report.created_at}",
            "",
        ]

        for section in report.sections:

            lines.append(
                section.title.upper()
            )

            lines.append(
                "-" * len(
                    section.title
                )
            )

            if section.content:

                lines.append(
                    section.content
                )

            for bullet in section.bullets:

                lines.append(
                    f"• {bullet}"
                )

            lines.append("")

        return "\n".join(
            lines
        ).strip()

    # ============================================================
    # HTML
    # ============================================================

    def to_html(
        self,
        report: ResearchReport,
    ) -> str:

        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            (
                f"<title>"
                f"{self._escape_html(report.title)}"
                f"</title>"
            ),
            "<style>",
            "body {",
            "font-family: Arial, sans-serif;",
            "max-width: 900px;",
            "margin: 40px auto;",
            "padding: 0 20px;",
            "line-height: 1.6;",
            "}",
            "h1 { margin-bottom: 5px; }",
            "h2 { margin-top: 35px; }",
            "li { margin-bottom: 6px; }",
            ".meta { color: #666; }",
            "</style>",
            "</head>",
            "<body>",
            (
                f"<h1>"
                f"{self._escape_html(report.title)}"
                f"</h1>"
            ),
            (
                f'<p class="meta">'
                f"Topic: "
                f"{self._escape_html(report.topic)}"
                f"<br>"
                f"Generated: "
                f"{self._escape_html(report.created_at)}"
                f"</p>"
            ),
        ]

        for section in report.sections:

            heading_level = max(
                2,
                min(
                    6,
                    section.level + 1,
                ),
            )

            html.append(
                f"<h{heading_level}>"
                f"{self._escape_html(section.title)}"
                f"</h{heading_level}>"
            )

            if section.content:

                paragraphs = (
                    section.content
                    .split("\n\n")
                )

                for paragraph in paragraphs:

                    if paragraph.strip():

                        html.append(
                            "<p>"
                            + self._escape_html(
                                paragraph
                            )
                            + "</p>"
                        )

            if section.bullets:

                html.append(
                    "<ul>"
                )

                for bullet in section.bullets:

                    html.append(
                        "<li>"
                        + self._escape_html(
                            bullet
                        )
                        + "</li>"
                    )

                html.append(
                    "</ul>"
                )

        html.extend(
            [
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(
            html
        )

    # ============================================================
    # FILE EXPORT
    # ============================================================

    def save_markdown(
        self,
        report: ResearchReport,
        path: str,
    ) -> str:

        content = self.to_markdown(
            report
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                content
            )

        return path

    def save_text(
        self,
        report: ResearchReport,
        path: str,
    ) -> str:

        content = self.to_text(
            report
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                content
            )

        return path

    def save_html(
        self,
        report: ResearchReport,
        path: str,
    ) -> str:

        content = self.to_html(
            report
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                content
            )

        return path

    # ============================================================
    # REPORT UTILITIES
    # ============================================================

    def add_section(
        self,
        report: ResearchReport,
        title: str,
        *,
        content: str = "",
        bullets: Iterable[str] | None = None,
        level: int = 1,
    ) -> ResearchReport:

        report.sections.append(
            ReportSection(
                title=self._clean_text(
                    title
                ),
                content=self._clean_text(
                    content
                ),
                bullets=[
                    self._clean_text(
                        bullet
                    )
                    for bullet in (
                        bullets or []
                    )
                    if self._clean_text(
                        bullet
                    )
                ],
                level=level,
            )
        )

        report.metadata[
            "section_count"
        ] = len(
            report.sections
        )

        return report

    def word_count(
        self,
        report: ResearchReport,
    ) -> int:

        text = self.to_text(
            report
        )

        return len(
            re.findall(
                r"\b\w+\b",
                text,
            )
        )

    def section(
        self,
        report: ResearchReport,
        title: str,
    ) -> ReportSection | None:

        normalized = (
            self._normalize(
                title
            )
        )

        for section in report.sections:

            if (
                self._normalize(
                    section.title
                )
                == normalized
            ):
                return section

        return None

    # ============================================================
    # TITLE
    # ============================================================

    def _generate_title(
        self,
        topic: str,
    ) -> str:

        if not topic:
            return self.default_title

        return (
            f"Research Report: "
            f"{topic}"
        )

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            str(
                value
            ).strip(),
        )

    @staticmethod
    def _normalize(
        value: Any,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            str(
                value or ""
            ).strip().lower(),
        )

    @staticmethod
    def _get(
        source: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if isinstance(
            source,
            dict,
        ):
            return source.get(
                key,
                default,
            )

        return getattr(
            source,
            key,
            default,
        )

    @staticmethod
    def _escape_html(
        value: Any,
    ) -> str:

        text = str(
            value or ""
        )

        return (
            text.replace(
                "&",
                "&amp;",
            )
            .replace(
                "<",
                "&lt;",
            )
            .replace(
                ">",
                "&gt;",
            )
            .replace(
                '"',
                "&quot;",
            )
            .replace(
                "'",
                "&#39;",
            )
        )


__all__ = [
    "ReportSection",
    "ResearchReport",
    "ReportGenerator",
]


