"""
RENIX Web Search
================

Browser-level web search abstraction.

Provides:
    - Search engine queries
    - Result extraction
    - Result ranking
    - Search history
    - Multiple search-engine support
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Iterable
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Normalized web search result."""

    title: str
    url: str
    snippet: str = ""
    source: str = ""
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WebSearch:
    """Perform and normalize browser-based web searches."""

    DEFAULT_ENGINES = {
        "google": "https://www.google.com/search?q={query}",
        "bing": "https://www.bing.com/search?q={query}",
        "duckduckgo": "https://duckduckgo.com/?q={query}",
    }

    def __init__(
        self,
        browser_manager: Any = None,
        *,
        default_engine: str = "google",
        engines: dict[str, str] | None = None,
        max_results: int = 10,
        request_delay: float = 0.0,
    ) -> None:

        self.browser_manager = browser_manager

        self.engines = dict(
            self.DEFAULT_ENGINES
        )

        if engines:
            self.engines.update(
                engines
            )

        self.default_engine = (
            default_engine.lower()
        )

        if self.default_engine not in self.engines:
            raise ValueError(
                f"Unknown search engine: "
                f"{self.default_engine}"
            )

        self.max_results = max(
            1,
            int(max_results),
        )

        self.request_delay = max(
            0.0,
            float(request_delay),
        )

        self.history: list[
            dict[str, Any]
        ] = []

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        *,
        engine: str | None = None,
        max_results: int | None = None,
        page_id: str | None = None,
        open_results: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search the web and return normalized results.

        Args:
            query:
                Search query.

            engine:
                Search engine name.

            max_results:
                Maximum results to return.

            page_id:
                Existing browser page.

            open_results:
                If True, search results are opened in the browser.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        engine_name = (
            engine.lower()
            if engine
            else self.default_engine
        )

        if engine_name not in self.engines:
            raise ValueError(
                f"Unsupported search engine: "
                f"{engine_name}"
            )

        limit = (
            max_results
            if max_results is not None
            else self.max_results
        )

        limit = max(
            1,
            int(limit),
        )

        url = self.build_search_url(
            query,
            engine=engine_name,
        )

        if self.request_delay:
            time.sleep(
                self.request_delay
            )

        actual_page_id = page_id

        if self.browser_manager is None:
            raise RuntimeError(
                "BrowserManager is not configured."
            )

        if actual_page_id is None:

            actual_page_id = (
                self.browser_manager.ensure_page()
            )

        self.browser_manager.navigate(
            url,
            page_id=actual_page_id,
        )

        results = self.extract_results(
            page_id=actual_page_id,
            engine=engine_name,
            max_results=limit,
        )

        history_entry = {
            "query": query,
            "engine": engine_name,
            "url": url,
            "results_count": len(results),
            "timestamp": time.time(),
        }

        self.history.append(
            history_entry
        )

        if open_results:
            return results

        return results

    # ========================================================
    # URL
    # ========================================================

    def build_search_url(
        self,
        query: str,
        *,
        engine: str | None = None,
    ) -> str:

        engine_name = (
            engine.lower()
            if engine
            else self.default_engine
        )

        template = self.engines.get(
            engine_name
        )

        if template is None:
            raise ValueError(
                f"Unknown search engine: "
                f"{engine_name}"
            )

        return template.format(
            query=quote_plus(
                query
            )
        )

    # ========================================================
    # RESULT EXTRACTION
    # ========================================================

    def extract_results(
        self,
        *,
        page_id: str | None = None,
        engine: str | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract search results from the current page."""

        if self.browser_manager is None:
            raise RuntimeError(
                "BrowserManager is not configured."
            )

        page = self.browser_manager.get_page(
            page_id
        )

        engine_name = (
            engine.lower()
            if engine
            else self.default_engine
        )

        limit = (
            max_results
            if max_results is not None
            else self.max_results
        )

        raw_results = self._extract_engine_results(
            page,
            engine_name,
        )

        normalized = []

        for index, result in enumerate(
            raw_results[:limit],
            start=1,
        ):

            normalized.append(
                SearchResult(
                    title=str(
                        result.get(
                            "title",
                            "",
                        )
                    ).strip(),
                    url=str(
                        result.get(
                            "url",
                            "",
                        )
                    ).strip(),
                    snippet=str(
                        result.get(
                            "snippet",
                            "",
                        )
                    ).strip(),
                    source=engine_name,
                    rank=index,
                ).to_dict()
            )

        return [
            result
            for result in normalized
            if result["url"]
        ]

    def _extract_engine_results(
        self,
        page: Any,
        engine: str,
    ) -> list[dict[str, Any]]:

        if engine == "google":
            return self._extract_google(
                page
            )

        if engine == "bing":
            return self._extract_bing(
                page
            )

        if engine == "duckduckgo":
            return self._extract_duckduckgo(
                page
            )

        return self._extract_generic(
            page
        )

    # ========================================================
    # GOOGLE
    # ========================================================

    def _extract_google(
        self,
        page: Any,
    ) -> list[dict[str, Any]]:

        script = """
        () => Array.from(
            document.querySelectorAll("div.MjjYud")
        ).map((container) => {
            const link = container.querySelector("a");
            const heading = container.querySelector("h3");

            if (!link || !heading) {
                return null;
            }

            const text = container.innerText || "";

            return {
                title: heading.innerText || "",
                url: link.href || "",
                snippet: text
            };
        }).filter(Boolean)
        """

        results = self._evaluate(
            page,
            script,
        )

        return self._clean_raw_results(
            results
        )

    # ========================================================
    # BING
    # ========================================================

    def _extract_bing(
        self,
        page: Any,
    ) -> list[dict[str, Any]]:

        script = """
        () => Array.from(
            document.querySelectorAll("li.b_algo")
        ).map((container) => {
            const link = container.querySelector("h2 a");
            const paragraph =
                container.querySelector(".b_caption p");

            if (!link) {
                return null;
            }

            return {
                title: link.innerText || "",
                url: link.href || "",
                snippet: paragraph
                    ? paragraph.innerText || ""
                    : ""
            };
        }).filter(Boolean)
        """

        results = self._evaluate(
            page,
            script,
        )

        return self._clean_raw_results(
            results
        )

    # ========================================================
    # DUCKDUCKGO
    # ========================================================

    def _extract_duckduckgo(
        self,
        page: Any,
    ) -> list[dict[str, Any]]:

        script = """
        () => Array.from(
            document.querySelectorAll(
                '[data-testid="result"]'
            )
        ).map((container) => {
            const link =
                container.querySelector("a");

            const heading =
                container.querySelector("h2");

            if (!link) {
                return null;
            }

            return {
                title: heading
                    ? heading.innerText || ""
                    : link.innerText || "",
                url: link.href || "",
                snippet: container.innerText || ""
            };
        }).filter(Boolean)
        """

        results = self._evaluate(
            page,
            script,
        )

        return self._clean_raw_results(
            results
        )

    # ========================================================
    # GENERIC
    # ========================================================

    def _extract_generic(
        self,
        page: Any,
    ) -> list[dict[str, Any]]:

        script = """
        () => Array.from(
            document.querySelectorAll("a[href]")
        ).map((link) => ({
            title: (link.innerText || "").trim(),
            url: link.href || "",
            snippet: ""
        })).filter(
            item =>
                item.title &&
                item.url &&
                /^https?:/.test(item.url)
        )
        """

        results = self._evaluate(
            page,
            script,
        )

        return self._clean_raw_results(
            results
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    @staticmethod
    def _evaluate(
        page: Any,
        script: str,
    ) -> list[dict[str, Any]]:

        evaluator = getattr(
            page,
            "evaluate",
            None,
        )

        if not callable(evaluator):
            raise RuntimeError(
                "Browser page does not support evaluation."
            )

        result = evaluator(
            script
        )

        if not isinstance(
            result,
            list,
        ):
            return []

        return result

    # ========================================================
    # CLEANING
    # ========================================================

    @staticmethod
    def _clean_raw_results(
        results: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        cleaned = []

        seen_urls: set[str] = set()

        for item in results:

            if not isinstance(
                item,
                dict,
            ):
                continue

            title = WebSearch._clean_string(
                item.get(
                    "title",
                    "",
                )
            )

            url = WebSearch._clean_url(
                item.get(
                    "url",
                    "",
                )
            )

            snippet = WebSearch._clean_string(
                item.get(
                    "snippet",
                    "",
                )
            )

            if not url:
                continue

            if url in seen_urls:
                continue

            seen_urls.add(
                url
            )

            cleaned.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

        return cleaned

    @staticmethod
    def _clean_string(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        text = str(
            value
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _clean_url(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        url = str(
            value
        ).strip()

        if not re.match(
            r"^https?://",
            url,
            re.IGNORECASE,
        ):
            return ""

        return url

    # ========================================================
    # HISTORY
    # ========================================================

    def get_history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:

        history = list(
            self.history
        )

        if limit is None:
            return history

        return history[
            -max(
                0,
                int(limit),
            ):
        ]

    def clear_history(self) -> None:
        self.history.clear()

    # ========================================================
    # ENGINE MANAGEMENT
    # ========================================================

    def add_engine(
        self,
        name: str,
        url_template: str,
    ) -> None:

        name = name.strip().lower()

        if not name:
            raise ValueError(
                "Engine name cannot be empty."
            )

        if "{query}" not in url_template:
            raise ValueError(
                "Search URL must contain {query}."
            )

        self.engines[
            name
        ] = url_template

    def remove_engine(
        self,
        name: str,
    ) -> bool:

        name = name.lower()

        if name not in self.engines:
            return False

        if name in self.DEFAULT_ENGINES:
            raise ValueError(
                "Built-in search engines cannot be removed."
            )

        del self.engines[
            name
        ]

        return True

    def set_default_engine(
        self,
        name: str,
    ) -> None:

        name = name.lower()

        if name not in self.engines:
            raise ValueError(
                f"Unknown search engine: {name}"
            )

        self.default_engine = name


__all__ = [
    "SearchResult",
    "WebSearch",
]


