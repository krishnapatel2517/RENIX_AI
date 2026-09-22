"""
RENIX AI - Research Agent

Handles research-oriented tasks for RENIX.

Responsibilities:
- Research topics
- Gather and organize information
- Compare sources
- Fact-check information
- Summarize research
- Generate structured reports
- Prepare citations
- Rank sources
- Maintain research context

The agent is designed to work with RENIX's dedicated research
modules when they are available, while also providing safe
fallback functionality when those modules are still being built.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .base_agent import (
    AgentCapability,
    AgentPriority,
    AgentResult,
    AgentTask,
    BaseAgent,
)


class ResearchAgent(BaseAgent):
    """
    RENIX research and information-analysis agent.

    This agent coordinates the research subsystem and exposes a
    single interface to the rest of RENIX.
    """

    agent_name = "research_agent"

    agent_description = (
        "Researches topics, evaluates information, compares sources, "
        "fact-checks claims, summarizes findings and generates reports."
    )

    agent_version = "1.0.0"

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        priority: AgentPriority = AgentPriority.NORMAL,
        logger: Optional[logging.Logger] = None,
        event_callback=None,
        confirmation_callback=None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            priority=priority,
            logger=logger,
            event_callback=event_callback,
            confirmation_callback=confirmation_callback,
            context=context,
        )

        self.research_engine = None
        self.source_manager = None
        self.source_ranker = None
        self.fact_checker = None
        self.summarizer = None
        self.comparison_engine = None
        self.citation_manager = None
        self.report_generator = None

        self._modules_loaded = False

        self._research_sessions: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self.default_max_sources = 10
        self.default_max_results = 20

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def _register_default_capabilities(self) -> None:
        """Register research capabilities."""

        super()._register_default_capabilities()

        capabilities = [
            AgentCapability(
                name="research",
                description=(
                    "Conduct research on a specified topic."
                ),
            ),
            AgentCapability(
                name="find_sources",
                description=(
                    "Find and organize relevant information sources."
                ),
            ),
            AgentCapability(
                name="rank_sources",
                description=(
                    "Rank sources by relevance and reliability."
                ),
            ),
            AgentCapability(
                name="fact_check",
                description=(
                    "Evaluate a claim against available evidence."
                ),
            ),
            AgentCapability(
                name="summarize",
                description=(
                    "Summarize research material."
                ),
            ),
            AgentCapability(
                name="compare",
                description=(
                    "Compare multiple topics, sources or options."
                ),
            ),
            AgentCapability(
                name="citations",
                description=(
                    "Create structured citations for sources."
                ),
            ),
            AgentCapability(
                name="report",
                description=(
                    "Generate a structured research report."
                ),
            ),
        ]

        for capability in capabilities:
            self.register_capability(
                capability
            )

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    async def on_initialize(self) -> bool:
        """
        Initialize the research subsystem.
        """

        self._load_research_modules()

        return True

    def _load_research_modules(self) -> None:
        """
        Load dedicated RENIX research modules.

        Missing modules do not prevent the agent from starting because
        the project is being built incrementally.
        """

        if self._modules_loaded:
            return

        self._modules_loaded = True

        module_map = {
            "research_engine": (
                "research.research_engine",
                "ResearchEngine",
            ),
            "source_manager": (
                "research.source_manager",
                "SourceManager",
            ),
            "source_ranker": (
                "research.source_ranker",
                "SourceRanker",
            ),
            "fact_checker": (
                "research.fact_checker",
                "FactChecker",
            ),
            "summarizer": (
                "research.summarizer",
                "Summarizer",
            ),
            "comparison_engine": (
                "research.comparison_engine",
                "ComparisonEngine",
            ),
            "citation_manager": (
                "research.citation_manager",
                "CitationManager",
            ),
            "report_generator": (
                "research.report_generator",
                "ReportGenerator",
            ),
        }

        for attribute, (
            module_name,
            class_name,
        ) in module_map.items():

            try:

                module = __import__(
                    module_name,
                    fromlist=[class_name],
                )

                cls = getattr(
                    module,
                    class_name,
                )

                try:
                    instance = cls()
                except TypeError:
                    instance = cls

                setattr(
                    self,
                    attribute,
                    instance,
                )

            except Exception as exc:

                self.logger.debug(
                    "Research module %s unavailable: %s",
                    class_name,
                    exc,
                )

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    async def execute(
        self,
        task: AgentTask,
    ) -> Any:
        """
        Execute a research task.
        """

        action = self._resolve_action(
            task
        )

        handlers = {
            "research": self.research,
            "search": self.research,
            "find_sources": self.find_sources,
            "sources": self.find_sources,
            "rank_sources": self.rank_sources,
            "rank": self.rank_sources,
            "fact_check": self.fact_check,
            "factcheck": self.fact_check,
            "verify": self.fact_check,
            "summarize": self.summarize,
            "summary": self.summarize,
            "compare": self.compare,
            "comparison": self.compare,
            "citations": self.create_citations,
            "citation": self.create_citations,
            "report": self.generate_report,
            "generate_report": self.generate_report,
            "session": self.get_session,
        }

        handler = handlers.get(
            action
        )

        if handler is None:

            return self.failure_result(
                task,
                f"Unknown research action: {action}",
                message=(
                    f"I don't know how to perform "
                    f"research action '{action}'."
                ),
                started_at=task.created_at,
            )

        try:

            parameters = dict(
                task.parameters
            )

            parameters.pop(
                "action",
                None,
            )

            parameters.pop(
                "capability",
                None,
            )

            result = handler(
                **parameters
            )

            if asyncio.iscoroutine(
                result
            ):
                result = await result

            return result

        except Exception as exc:

            self.logger.exception(
                "Research action failed: %s",
                action,
            )

            return self.failure_result(
                task,
                str(exc),
                message=(
                    f"Research action '{action}' failed."
                ),
                started_at=task.created_at,
            )

    # ========================================================================
    # ACTION RESOLUTION
    # ========================================================================

    def _resolve_action(
        self,
        task: AgentTask,
    ) -> str:

        explicit_action = task.parameters.get(
            "action"
        )

        if explicit_action:

            return str(
                explicit_action
            ).strip().lower()

        instruction = (
            task.instruction
            .strip()
            .lower()
        )

        if (
            instruction.startswith("research")
            or "research this" in instruction
            or "research about" in instruction
            or "research on" in instruction
        ):
            return "research"

        if (
            "find sources" in instruction
            or "find information" in instruction
            or "find references" in instruction
        ):
            return "find_sources"

        if (
            "rank sources" in instruction
            or "rank the sources" in instruction
        ):
            return "rank_sources"

        if (
            "fact check" in instruction
            or "fact-check" in instruction
            or "verify this claim" in instruction
            or "verify the claim" in instruction
        ):
            return "fact_check"

        if (
            "summarize" in instruction
            or "summarise" in instruction
            or "give me a summary" in instruction
        ):
            return "summarize"

        if (
            "compare" in instruction
            or "comparison" in instruction
            or "compare these" in instruction
        ):
            return "compare"

        if (
            "citation" in instruction
            or "citations" in instruction
            or "references" in instruction
        ):
            return "citations"

        if (
            "research report" in instruction
            or "generate report" in instruction
            or "research report" in instruction
        ):
            return "report"

        return "unknown"

    # ========================================================================
    # RESEARCH
    # ========================================================================

    async def research(
        self,
        topic: str,
        questions: Optional[
            Sequence[str]
        ] = None,
        max_sources: int = 10,
        depth: str = "standard",
        session_id: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Conduct research on a topic.

        The actual external retrieval mechanism can be connected later
        through the research engine or source manager.
        """

        if not topic or not topic.strip():

            return {
                "success": False,
                "error": (
                    "Research topic cannot be empty."
                ),
            }

        topic = topic.strip()

        session_id = (
            session_id
            or self._create_session_id(
                topic
            )
        )

        session = self._get_or_create_session(
            session_id,
            topic,
        )

        session["depth"] = depth

        if questions:

            session["questions"] = list(
                questions
            )

        if self.research_engine is not None:

            result = await self._call_module(
                self.research_engine,
                (
                    "research",
                    "run",
                    "execute",
                    "search",
                ),
                topic=topic,
                questions=questions,
                max_sources=max_sources,
                depth=depth,
                session_id=session_id,
            )

            if result is not None:

                normalized = self._normalize_research_result(
                    result,
                    topic,
                    session_id,
                )

                session["last_result"] = normalized

                return normalized

        sources = await self.find_sources(
            topic=topic,
            max_sources=max_sources,
            session_id=session_id,
        )

        source_items = sources.get(
            "sources",
            [],
        )

        session["sources"] = source_items

        summary = await self.summarize(
            text=self._sources_to_text(
                source_items
            ),
            topic=topic,
            session_id=session_id,
        )

        result = {
            "success": True,
            "action": "research",
            "topic": topic,
            "session_id": session_id,
            "depth": depth,
            "questions": list(
                questions or []
            ),
            "sources": source_items,
            "summary": summary.get(
                "summary",
                "",
            ),
            "note": (
                "The research engine is not yet connected "
                "to an external retrieval provider."
            ),
            "timestamp": self._timestamp(),
        }

        session["last_result"] = result

        return result

    # ========================================================================
    # SOURCE DISCOVERY
    # ========================================================================

    async def find_sources(
        self,
        topic: str,
        max_sources: int = 10,
        source_types: Optional[
            Sequence[str]
        ] = None,
        session_id: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Find sources for a research topic.

        When the dedicated source manager exists, it is used.
        Otherwise a structured placeholder result is returned rather
        than pretending that unsupported web retrieval occurred.
        """

        if not topic or not topic.strip():

            return {
                "success": False,
                "error": (
                    "Topic cannot be empty."
                ),
            }

        topic = topic.strip()

        max_sources = max(
            1,
            min(
                int(max_sources),
                100,
            ),
        )

        if self.source_manager is not None:

            result = await self._call_module(
                self.source_manager,
                (
                    "find_sources",
                    "search",
                    "find",
                    "get_sources",
                ),
                topic=topic,
                max_sources=max_sources,
                source_types=source_types,
            )

            if result is not None:

                sources = self._normalize_sources(
                    result
                )

                if session_id:

                    session = self._get_or_create_session(
                        session_id,
                        topic,
                    )

                    session["sources"] = sources

                return {
                    "success": True,
                    "action": "find_sources",
                    "topic": topic,
                    "sources": sources,
                    "count": len(sources),
                }

        sources: List[Dict[str, Any]] = []

        return {
            "success": True,
            "action": "find_sources",
            "topic": topic,
            "sources": sources,
            "count": 0,
            "note": (
                "No external source provider is currently "
                "connected to the research subsystem."
            ),
        }

    # ========================================================================
    # SOURCE RANKING
    # ========================================================================

    async def rank_sources(
        self,
        sources: Sequence[Any],
        topic: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Rank supplied sources.
        """

        source_list = list(
            sources or []
        )

        if self.source_ranker is not None:

            result = await self._call_module(
                self.source_ranker,
                (
                    "rank",
                    "rank_sources",
                    "score",
                    "rank_sources_by_relevance",
                ),
                sources=source_list,
                topic=topic,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "rank_sources",
                    "topic": topic,
                    "sources": self._normalize_sources(
                        result
                    ),
                }

        ranked = []

        for index, source in enumerate(
            source_list
        ):

            normalized = self._normalize_source(
                source
            )

            score = self._source_score(
                normalized,
                topic,
            )

            normalized["score"] = score

            ranked.append(
                normalized
            )

        ranked.sort(
            key=lambda item: item.get(
                "score",
                0,
            ),
            reverse=True,
        )

        return {
            "success": True,
            "action": "rank_sources",
            "topic": topic,
            "sources": ranked,
            "count": len(ranked),
        }

    def _source_score(
        self,
        source: Dict[str, Any],
        topic: Optional[str],
    ) -> float:

        score = 0.0

        url = str(
            source.get(
                "url",
                "",
            )
        ).lower()

        title = str(
            source.get(
                "title",
                "",
            )
        ).lower()

        text = str(
            source.get(
                "description",
                "",
            )
        ).lower()

        if url.startswith(
            "https://"
        ):

            score += 0.15

        trusted_domains = (
            ".gov",
            ".edu",
            ".ac.",
            "who.int",
            "un.org",
            "worldbank.org",
            "nature.com",
            "science.org",
            "ieee.org",
        )

        if any(
            domain in url
            for domain in trusted_domains
        ):

            score += 0.45

        if topic:

            topic_words = self._tokenize(
                topic
            )

            combined = (
                title
                + " "
                + text
            )

            matched = sum(
                1
                for word in topic_words
                if word in combined
            )

            if topic_words:

                score += min(
                    0.4,
                    matched
                    / len(topic_words)
                    * 0.4,
                )

        return round(
            min(
                1.0,
                score,
            ),
            3,
        )

    # ========================================================================
    # FACT CHECKING
    # ========================================================================

    async def fact_check(
        self,
        claim: str,
        evidence: Optional[
            Sequence[Any]
        ] = None,
        topic: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate a claim against supplied evidence.
        """

        if not claim or not claim.strip():

            return {
                "success": False,
                "error": (
                    "Claim cannot be empty."
                ),
            }

        claim = claim.strip()

        if self.fact_checker is not None:

            result = await self._call_module(
                self.fact_checker,
                (
                    "fact_check",
                    "check",
                    "verify",
                    "validate",
                ),
                claim=claim,
                evidence=evidence,
                topic=topic,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "fact_check",
                    "claim": claim,
                    "result": result,
                }

        evidence_list = list(
            evidence or []
        )

        if not evidence_list:

            return {
                "success": True,
                "action": "fact_check",
                "claim": claim,
                "verdict": "insufficient_evidence",
                "confidence": 0.0,
                "evidence": [],
                "reason": (
                    "No evidence was supplied and no "
                    "fact-checking provider is connected."
                ),
            }

        normalized_evidence = [
            self._normalize_source(
                item
            )
            for item in evidence_list
        ]

        supporting = 0
        contradicting = 0

        claim_tokens = set(
            self._tokenize(
                claim
            )
        )

        for item in normalized_evidence:

            text = (
                str(
                    item.get(
                        "title",
                        "",
                    )
                )
                + " "
                + str(
                    item.get(
                        "description",
                        "",
                    )
                )
            ).lower()

            if not text:
                continue

            overlap = len(
                claim_tokens.intersection(
                    set(
                        self._tokenize(
                            text
                        )
                    )
                )
            )

            if overlap > 0:

                supporting += 1

        if supporting == 0:

            verdict = "uncertain"

        elif supporting == len(
            normalized_evidence
        ):

            verdict = "supported"

        else:

            verdict = "partially_supported"

        confidence = (
            supporting
            / len(
                normalized_evidence
            )
        )

        return {
            "success": True,
            "action": "fact_check",
            "claim": claim,
            "verdict": verdict,
            "confidence": round(
                confidence,
                3,
            ),
            "evidence": normalized_evidence,
            "reason": (
                "Fallback evidence matching was used. "
                "This is not equivalent to independent fact verification."
            ),
        }

    # ========================================================================
    # SUMMARIZATION
    # ========================================================================

    async def summarize(
        self,
        text: Optional[str] = None,
        sources: Optional[
            Sequence[Any]
        ] = None,
        topic: Optional[str] = None,
        max_length: int = 1200,
        session_id: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Summarize research material.
        """

        if sources:

            text = self._sources_to_text(
                sources
            )

        text = (
            text
            or ""
        ).strip()

        if not text:

            return {
                "success": False,
                "error": (
                    "No text or sources were supplied."
                ),
            }

        if self.summarizer is not None:

            result = await self._call_module(
                self.summarizer,
                (
                    "summarize",
                    "summarise",
                    "create_summary",
                ),
                text=text,
                topic=topic,
                max_length=max_length,
            )

            if result is not None:

                summary_text = self._extract_summary(
                    result
                )

                if session_id:

                    session = self._get_or_create_session(
                        session_id,
                        topic or "",
                    )

                    session["summary"] = summary_text

                return {
                    "success": True,
                    "action": "summarize",
                    "topic": topic,
                    "summary": summary_text,
                    "result": result,
                }

        summary = self._fallback_summary(
            text,
            max_length=max_length,
        )

        if session_id:

            session = self._get_or_create_session(
                session_id,
                topic or "",
            )

            session["summary"] = summary

        return {
            "success": True,
            "action": "summarize",
            "topic": topic,
            "summary": summary,
            "method": "fallback",
        }

    def _fallback_summary(
        self,
        text: str,
        max_length: int = 1200,
    ) -> str:

        cleaned = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if len(cleaned) <= max_length:

            return cleaned

        sentences = re.split(
            r"(?<=[.!?])\s+",
            cleaned,
        )

        selected = []

        current_length = 0

        for sentence in sentences:

            if (
                current_length
                + len(sentence)
                + 1
                > max_length
            ):
                break

            selected.append(
                sentence
            )

            current_length += (
                len(sentence)
                + 1
            )

        if selected:

            return " ".join(
                selected
            )

        return (
            cleaned[:max_length]
            + "..."
        )

    # ========================================================================
    # COMPARISON
    # ========================================================================

    async def compare(
        self,
        items: Sequence[Any],
        criteria: Optional[
            Sequence[str]
        ] = None,
        topic: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Compare multiple items.
        """

        item_list = list(
            items or []
        )

        if len(item_list) < 2:

            return {
                "success": False,
                "error": (
                    "At least two items are required for comparison."
                ),
            }

        criteria_list = list(
            criteria or []
        )

        if self.comparison_engine is not None:

            result = await self._call_module(
                self.comparison_engine,
                (
                    "compare",
                    "comparison",
                    "run",
                ),
                items=item_list,
                criteria=criteria_list,
                topic=topic,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "compare",
                    "topic": topic,
                    "result": result,
                }

        normalized_items = [
            self._normalize_source(
                item
            )
            if isinstance(
                item,
                (dict, str),
            )
            else {
                "name": str(
                    item
                )
            }
            for item in item_list
        ]

        comparison = []

        for item in normalized_items:

            row = {
                "name": (
                    item.get(
                        "name"
                    )
                    or item.get(
                        "title"
                    )
                    or "Unnamed",
                )
            }

            for criterion in criteria_list:

                row[criterion] = item.get(
                    criterion,
                    "Not available",
                )

            comparison.append(
                row
            )

        return {
            "success": True,
            "action": "compare",
            "topic": topic,
            "criteria": criteria_list,
            "items": comparison,
            "count": len(comparison),
            "method": "fallback",
        }

    # ========================================================================
    # CITATIONS
    # ========================================================================

    async def create_citations(
        self,
        sources: Sequence[Any],
        style: str = "apa",
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Generate citations for sources.
        """

        source_list = list(
            sources or []
        )

        if self.citation_manager is not None:

            result = await self._call_module(
                self.citation_manager,
                (
                    "create_citations",
                    "cite",
                    "generate",
                ),
                sources=source_list,
                style=style,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "citations",
                    "style": style,
                    "citations": result,
                }

        citations = []

        for index, source in enumerate(
            source_list,
            start=1,
        ):

            normalized = self._normalize_source(
                source
            )

            citation = self._format_citation(
                normalized,
                style,
            )

            citations.append(
                {
                    "index": index,
                    "citation": citation,
                }
            )

        return {
            "success": True,
            "action": "citations",
            "style": style,
            "citations": citations,
        }

    def _format_citation(
        self,
        source: Dict[str, Any],
        style: str,
    ) -> str:

        title = str(
            source.get(
                "title",
                "Untitled source",
            )
        )

        author = str(
            source.get(
                "author",
                source.get(
                    "authors",
                    "Unknown author",
                ),
            )
        )

        year = str(
            source.get(
                "year",
                source.get(
                    "date",
                    "n.d.",
                ),
            )
        )

        url = str(
            source.get(
                "url",
                "",
            )
        )

        style = style.lower()

        if style == "mla":

            citation = (
                f"{author}. "
                f"\"{title}.\" "
                f"{year}."
            )

        elif style == "chicago":

            citation = (
                f"{author}. "
                f"\"{title}.\" "
                f"{year}."
            )

        else:

            citation = (
                f"{author}. "
                f"({year}). "
                f"{title}."
            )

        if url:

            citation += f" {url}"

        return citation

    # ========================================================================
    # REPORT GENERATION
    # ========================================================================

    async def generate_report(
        self,
        topic: str,
        sources: Optional[
            Sequence[Any]
        ] = None,
        summary: Optional[str] = None,
        findings: Optional[
            Sequence[str]
        ] = None,
        title: Optional[str] = None,
        citation_style: str = "apa",
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Generate a structured research report.
        """

        if not topic or not topic.strip():

            return {
                "success": False,
                "error": (
                    "Report topic cannot be empty."
                ),
            }

        topic = topic.strip()

        source_list = list(
            sources or []
        )

        if self.report_generator is not None:

            result = await self._call_module(
                self.report_generator,
                (
                    "generate_report",
                    "generate",
                    "create",
                ),
                topic=topic,
                sources=source_list,
                summary=summary,
                findings=findings,
                title=title,
                citation_style=citation_style,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "report",
                    "topic": topic,
                    "report": result,
                }

        if summary is None and source_list:

            summary_result = await self.summarize(
                sources=source_list,
                topic=topic,
            )

            summary = summary_result.get(
                "summary",
                "",
            )

        report_title = (
            title
            or f"Research Report: {topic}"
        )

        lines = [
            f"# {report_title}",
            "",
            f"**Generated:** {self._timestamp()}",
            "",
            "## Topic",
            "",
            topic,
            "",
        ]

        if summary:

            lines.extend(
                [
                    "## Summary",
                    "",
                    summary,
                    "",
                ]
            )

        if findings:

            lines.extend(
                [
                    "## Key Findings",
                    "",
                ]
            )

            for finding in findings:

                lines.append(
                    f"- {finding}"
                )

            lines.append("")

        if source_list:

            citation_result = await self.create_citations(
                source_list,
                style=citation_style,
            )

            lines.extend(
                [
                    "## Sources",
                    "",
                ]
            )

            for citation in citation_result.get(
                "citations",
                [],
            ):

                lines.append(
                    f"{citation.get('index', '')}. "
                    f"{citation.get('citation', '')}"
                )

            lines.append("")

        lines.extend(
            [
                "## Research Note",
                "",
                (
                    "This report was assembled by RENIX. "
                    "Claims should be independently verified when "
                    "the research provider or evidence base is limited."
                ),
            ]
        )

        report = "\n".join(
            lines
        )

        return {
            "success": True,
            "action": "report",
            "topic": topic,
            "title": report_title,
            "report": report,
            "citation_style": citation_style,
        }

    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================

    def _create_session_id(
        self,
        topic: str,
    ) -> str:

        timestamp = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y%m%d%H%M%S%f"
            )
        )

        slug = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            topic,
        ).strip("_").lower()

        return (
            f"research_{slug[:40]}_{timestamp}"
        )

    def _get_or_create_session(
        self,
        session_id: str,
        topic: str,
    ) -> Dict[str, Any]:

        if session_id not in self._research_sessions:

            self._research_sessions[
                session_id
            ] = {
                "session_id": session_id,
                "topic": topic,
                "created_at": self._timestamp(),
                "sources": [],
                "questions": [],
                "summary": "",
                "last_result": None,
            }

        return self._research_sessions[
            session_id
        ]

    async def get_session(
        self,
        session_id: str,
        **_: Any,
    ) -> Dict[str, Any]:

        session = self._research_sessions.get(
            session_id
        )

        if session is None:

            return {
                "success": False,
                "error": (
                    f"Research session not found: {session_id}"
                ),
            }

        return {
            "success": True,
            "action": "session",
            "session": session,
        }

    # ========================================================================
    # MODULE HELPERS
    # ========================================================================

    async def _call_module(
        self,
        module: Any,
        method_names: Iterable[str],
        **kwargs: Any,
    ) -> Any:
        """
        Try multiple compatible method names on a subsystem.
        """

        for method_name in method_names:

            method = getattr(
                module,
                method_name,
                None,
            )

            if not callable(
                method
            ):
                continue

            try:

                result = method(
                    **kwargs
                )

                if asyncio.iscoroutine(
                    result
                ):
                    result = await result

                return result

            except TypeError:

                try:

                    result = method(
                        kwargs
                    )

                    if asyncio.iscoroutine(
                        result
                    ):
                        result = await result

                    return result

                except Exception as exc:

                    self.logger.debug(
                        "Research method %s failed: %s",
                        method_name,
                        exc,
                    )

            except Exception as exc:

                self.logger.debug(
                    "Research method %s failed: %s",
                    method_name,
                    exc,
                )

        return None

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    def _normalize_research_result(
        self,
        result: Any,
        topic: str,
        session_id: str,
    ) -> Dict[str, Any]:

        if isinstance(
            result,
            dict,
        ):

            normalized = dict(
                result
            )

        else:

            normalized = {
                "result": result
            }

        normalized.setdefault(
            "success",
            True,
        )

        normalized.setdefault(
            "action",
            "research",
        )

        normalized.setdefault(
            "topic",
            topic,
        )

        normalized.setdefault(
            "session_id",
            session_id,
        )

        normalized.setdefault(
            "timestamp",
            self._timestamp(),
        )

        return normalized

    def _normalize_sources(
        self,
        result: Any,
    ) -> List[Dict[str, Any]]:

        if isinstance(
            result,
            dict,
        ):

            candidates = (
                result.get(
                    "sources"
                )
                or result.get(
                    "results"
                )
                or result.get(
                    "items"
                )
                or []
            )

        elif isinstance(
            result,
            (list, tuple),
        ):

            candidates = result

        else:

            candidates = [
                result
            ]

        return [
            self._normalize_source(
                item
            )
            for item in candidates
        ]

    def _normalize_source(
        self,
        source: Any,
    ) -> Dict[str, Any]:

        if isinstance(
            source,
            dict,
        ):

            return dict(
                source
            )

        if isinstance(
            source,
            str,
        ):

            return {
                "title": source,
                "url": "",
                "description": "",
            }

        return {
            "title": str(
                source
            ),
            "url": "",
            "description": "",
        }

    def _extract_summary(
        self,
        result: Any,
    ) -> str:

        if isinstance(
            result,
            str,
        ):

            return result

        if isinstance(
            result,
            dict,
        ):

            for key in (
                "summary",
                "text",
                "content",
                "result",
            ):

                value = result.get(
                    key
                )

                if value is not None:

                    return str(
                        value
                    )

        return str(
            result
        )

    def _sources_to_text(
        self,
        sources: Sequence[Any],
    ) -> str:

        parts = []

        for source in sources:

            normalized = self._normalize_source(
                source
            )

            title = normalized.get(
                "title",
                "",
            )

            description = normalized.get(
                "description",
                "",
            )

            url = normalized.get(
                "url",
                "",
            )

            parts.append(
                " ".join(
                    part
                    for part in (
                        str(title),
                        str(description),
                        str(url),
                    )
                    if part
                )
            )

        return "\n".join(
            parts
        )

    # ========================================================================
    # TEXT UTILITIES
    # ========================================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> List[str]:

        words = re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )

        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "into",
            "about",
            "what",
            "which",
            "where",
            "when",
            "how",
            "why",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "a",
            "an",
            "of",
            "to",
            "in",
            "on",
            "at",
            "by",
            "is",
            "it",
        }

        return [
            word
            for word in words
            if word not in stop_words
            and len(word) > 2
        ]

    @staticmethod
    def _timestamp() -> str:

        return (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        )


__all__ = [
    "ResearchAgent",
]


