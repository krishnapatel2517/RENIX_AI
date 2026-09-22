"""
RENIX AI - Browser Agent

Handles browser-related tasks for RENIX.

Responsibilities:
- Open websites
- Search the web
- Navigate to URLs
- Read webpages
- Extract links
- Manage browser tabs/windows where supported
- Download files
- Fill simple web forms where supported
- Provide a safe interface between RENIX and browser automation

The agent is designed to work even when optional browser automation
libraries are not installed. Optional integrations are detected at runtime.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urljoin, urlparse

from .base_agent import (
    AgentCapability,
    AgentPriority,
    AgentResult,
    AgentTask,
    BaseAgent,
)


class BrowserAgent(BaseAgent):
    """
    RENIX browser-management agent.

    Provides a unified browser interface for the AI orchestration layer.
    """

    agent_name = "browser_agent"

    agent_description = (
        "Controls web browsing, searches the internet, opens websites, "
        "reads webpages, extracts links and manages browser operations."
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

        self.browser_root = Path.cwd() / "browser"
        self.download_root = (
            Path.home() / "Downloads"
        )

        self.browser_manager = None
        self.browser_controller = None
        self.page_reader = None
        self.web_search = None
        self.webpage_parser = None
        self.form_controller = None
        self.link_handler = None
        self.download_manager = None
        self.browser_security = None

        self._modules_loaded = False
        self._playwright = None
        self._playwright_browser = None
        self._playwright_page = None

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def _register_default_capabilities(self) -> None:
        """Register browser capabilities."""

        super()._register_default_capabilities()

        capabilities = [
            AgentCapability(
                name="open_url",
                description="Open a website in the default browser.",
            ),
            AgentCapability(
                name="web_search",
                description="Search the web.",
            ),
            AgentCapability(
                name="navigate",
                description="Navigate to a URL.",
            ),
            AgentCapability(
                name="read_page",
                description="Read webpage content.",
            ),
            AgentCapability(
                name="extract_links",
                description="Extract links from a webpage.",
            ),
            AgentCapability(
                name="download",
                description="Download a file from a URL.",
            ),
            AgentCapability(
                name="browser_status",
                description="Check browser automation status.",
            ),
            AgentCapability(
                name="new_tab",
                description="Open a new browser tab.",
            ),
            AgentCapability(
                name="close_browser",
                description="Close RENIX-controlled browser.",
            ),
            AgentCapability(
                name="fill_form",
                description="Fill supported web forms.",
                requires_confirmation=True,
            ),
        ]

        for capability in capabilities:
            self.register_capability(capability)

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    async def on_initialize(self) -> bool:
        """
        Initialize browser subsystem.

        Browser automation is optional. RENIX can still open websites
        using the operating system browser when automation is unavailable.
        """

        self._load_browser_modules()

        return True

    def _load_browser_modules(self) -> None:
        """Load optional RENIX browser subsystem modules."""

        if self._modules_loaded:
            return

        self._modules_loaded = True

        module_map = {
            "browser_manager": (
                "browser.browser_manager",
                "BrowserManager",
            ),
            "browser_controller": (
                "browser.browser_controller",
                "BrowserController",
            ),
            "page_reader": (
                "browser.page_reader",
                "PageReader",
            ),
            "web_search": (
                "browser.web_search",
                "WebSearch",
            ),
            "webpage_parser": (
                "browser.webpage_parser",
                "WebpageParser",
            ),
            "form_controller": (
                "browser.form_controller",
                "FormController",
            ),
            "link_handler": (
                "browser.link_handler",
                "LinkHandler",
            ),
            "download_manager": (
                "browser.download_manager",
                "DownloadManager",
            ),
            "browser_security": (
                "browser.browser_security",
                "BrowserSecurity",
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

                setattr(
                    self,
                    attribute,
                    cls(),
                )

            except Exception as exc:

                self.logger.debug(
                    "%s unavailable: %s",
                    class_name,
                    exc,
                )

        self._load_playwright()

    def _load_playwright(self) -> None:
        """Detect Playwright without making it mandatory."""

        try:

            from playwright.async_api import (
                async_playwright,
            )

            self._playwright = async_playwright

            self.logger.debug(
                "Playwright detected."
            )

        except Exception:

            self._playwright = None

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    async def execute(
        self,
        task: AgentTask,
    ) -> Any:
        """
        Execute a browser task.
        """

        action = self._resolve_action(task)

        handlers = {
            "open_url": self.open_url,
            "navigate": self.navigate,
            "search": self.search_web,
            "web_search": self.search_web,
            "read": self.read_page,
            "read_page": self.read_page,
            "links": self.extract_links,
            "extract_links": self.extract_links,
            "download": self.download,
            "status": self.browser_status,
            "browser_status": self.browser_status,
            "new_tab": self.new_tab,
            "close": self.close_browser,
            "close_browser": self.close_browser,
            "fill_form": self.fill_form,
        }

        handler = handlers.get(
            action
        )

        if handler is None:

            return self.failure_result(
                task,
                f"Unknown browser action: {action}",
                message=(
                    f"I don't know how to perform browser "
                    f"action '{action}'."
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

            if asyncio.iscoroutine(result):
                result = await result

            return result

        except Exception as exc:

            self.logger.exception(
                "Browser action failed: %s",
                action,
            )

            return self.failure_result(
                task,
                str(exc),
                message=(
                    f"Browser action '{action}' failed."
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
            "search the web" in instruction
            or instruction.startswith("search")
            or "google" in instruction
            or "bing" in instruction
        ):
            return "search"

        if (
            "read webpage" in instruction
            or "read web page" in instruction
            or "read this page" in instruction
        ):
            return "read_page"

        if (
            "extract links" in instruction
            or "get links" in instruction
        ):
            return "extract_links"

        if (
            "download" in instruction
        ):
            return "download"

        if (
            "new tab" in instruction
            or "open tab" in instruction
        ):
            return "new_tab"

        if (
            "close browser" in instruction
            or "close browser" in instruction
        ):
            return "close_browser"

        if (
            "fill form" in instruction
            or "fill the form" in instruction
        ):
            return "fill_form"

        if (
            "browser status" in instruction
            or "is browser" in instruction
        ):
            return "browser_status"

        if (
            instruction.startswith("open ")
            or instruction.startswith("go to ")
            or instruction.startswith("visit ")
        ):
            return "open_url"

        if (
            "navigate" in instruction
        ):
            return "navigate"

        return "unknown"

    # ========================================================================
    # URL HELPERS
    # ========================================================================

    @staticmethod
    def _normalize_url(
        url: str,
    ) -> str:

        url = str(
            url
        ).strip()

        if not url:
            raise ValueError(
                "URL cannot be empty."
            )

        parsed = urlparse(
            url
        )

        if not parsed.scheme:

            if (
                "." in url
                and " " not in url
            ):

                url = (
                    "https://"
                    + url
                )

            else:

                url = (
                    "https://www.google.com/search?q="
                    + quote_plus(url)
                )

        return url

    @staticmethod
    def _is_http_url(
        url: str,
    ) -> bool:

        parsed = urlparse(
            url
        )

        return parsed.scheme in {
            "http",
            "https",
        }

    # ========================================================================
    # OPEN URL
    # ========================================================================

    async def open_url(
        self,
        url: str,
        browser: Optional[str] = None,
        new_window: bool = True,
        **_: Any,
    ) -> Dict[str, Any]:

        normalized_url = self._normalize_url(
            url
        )

        if not self._is_http_url(
            normalized_url
        ):

            return {
                "success": False,
                "error": (
                    "Only HTTP and HTTPS URLs are "
                    "supported."
                ),
            }

        # Try RENIX browser controller.

        if self.browser_controller is not None:

            for method_name in (
                "open_url",
                "open",
                "navigate",
            ):

                method = getattr(
                    self.browser_controller,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            normalized_url
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "open_url",
                            "url": normalized_url,
                            "result": result,
                        }

                    except Exception:
                        pass

        try:

            opened = webbrowser.open(
                normalized_url,
                new=1 if new_window else 0,
            )

            return {
                "success": bool(opened),
                "action": "open_url",
                "url": normalized_url,
                "browser": browser,
                "message": (
                    "Website opened in the default browser."
                    if opened
                    else "Browser open request was not accepted."
                ),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # NAVIGATION
    # ========================================================================

    async def navigate(
        self,
        url: str,
        **_: Any,
    ) -> Dict[str, Any]:

        normalized_url = self._normalize_url(
            url
        )

        if self._playwright_page is not None:

            try:

                response = await self._playwright_page.goto(
                    normalized_url,
                    wait_until="domcontentloaded",
                )

                return {
                    "success": True,
                    "action": "navigate",
                    "url": self._playwright_page.url,
                    "status": (
                        response.status
                        if response is not None
                        else None
                    ),
                }

            except Exception as exc:

                return {
                    "success": False,
                    "error": str(exc),
                }

        return await self._start_browser_and_navigate(
            normalized_url
        )

    async def _start_browser_and_navigate(
        self,
        url: str,
    ) -> Dict[str, Any]:

        if self._playwright is None:

            return await self.open_url(
                url
            )

        try:

            playwright_instance = (
                await self._playwright().start()
            )

            self._playwright_browser = (
                await playwright_instance.chromium.launch(
                    headless=False
                )
            )

            self._playwright_page = (
                await self._playwright_browser.new_page()
            )

            response = await self._playwright_page.goto(
                url,
                wait_until="domcontentloaded",
            )

            return {
                "success": True,
                "action": "navigate",
                "url": self._playwright_page.url,
                "status": (
                    response.status
                    if response is not None
                    else None
                ),
                "automation": True,
            }

        except Exception as exc:

            self.logger.warning(
                "Could not start Playwright: %s",
                exc,
            )

            return await self.open_url(
                url
            )

    # ========================================================================
    # WEB SEARCH
    # ========================================================================

    async def search_web(
        self,
        query: str,
        engine: str = "google",
        open_browser: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:

        if not query:

            return {
                "success": False,
                "error": (
                    "Search query cannot be empty."
                ),
            }

        # Use RENIX web-search subsystem if available.

        if self.web_search is not None:

            for method_name in (
                "search",
                "search_web",
                "query",
            ):

                method = getattr(
                    self.web_search,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            query=query,
                            engine=engine,
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "search",
                            "query": query,
                            "engine": engine,
                            "results": result,
                        }

                    except TypeError:

                        try:

                            result = method(
                                query
                            )

                            if asyncio.iscoroutine(
                                result
                            ):
                                result = await result

                            return {
                                "success": True,
                                "action": "search",
                                "query": query,
                                "engine": engine,
                                "results": result,
                            }

                        except Exception:
                            pass

                    except Exception:
                        pass

        engines = {
            "google": (
                "https://www.google.com/search?q="
            ),
            "bing": (
                "https://www.bing.com/search?q="
            ),
            "duckduckgo": (
                "https://duckduckgo.com/?q="
            ),
            "brave": (
                "https://search.brave.com/search?q="
            ),
        }

        base_url = engines.get(
            engine.lower(),
            engines["google"],
        )

        search_url = (
            base_url
            + quote_plus(query)
        )

        if open_browser:
            return await self.open_url(
                search_url
            )

        # Attempt actual search via duckduckgo_search if available
        results = []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                ddg_res = list(ddgs.text(query, max_results=5))
                for item in ddg_res:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("href") or item.get("link", ""),
                        "snippet": item.get("body") or item.get("snippet", ""),
                    })
        except Exception as exc:
            self.logger.debug("Direct DDGS search unavailable: %s", exc)

        return {
            "success": True,
            "action": "search",
            "query": query,
            "engine": engine,
            "results": results,
            "count": len(results),
            "search_url": search_url,
        }

    # ========================================================================
    # READ PAGE
    # ========================================================================

    async def read_page(
        self,
        url: Optional[str] = None,
        max_chars: int = 50_000,
        **_: Any,
    ) -> Dict[str, Any]:

        # Use existing page reader.

        if (
            self.page_reader is not None
            and url is not None
        ):

            for method_name in (
                "read",
                "read_page",
                "extract_text",
            ):

                method = getattr(
                    self.page_reader,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            url
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "read_page",
                            "url": url,
                            "content": self._truncate(
                                result,
                                max_chars,
                            ),
                        }

                    except Exception:
                        pass

        # Use active Playwright page.

        if (
            url is not None
            and self._playwright_page is None
        ):

            navigation = await self.navigate(
                url
            )

            if not navigation.get(
                "success"
            ):

                return navigation

        if self._playwright_page is not None:

            try:

                title = await self._playwright_page.title()

                text = await self._playwright_page.locator(
                    "body"
                ).inner_text()

                return {
                    "success": True,
                    "action": "read_page",
                    "url": self._playwright_page.url,
                    "title": title,
                    "content": self._truncate(
                        text,
                        max_chars,
                    ),
                }

            except Exception as exc:

                return {
                    "success": False,
                    "error": str(exc),
                }

        return {
            "success": False,
            "error": (
                "No browser automation session is available "
                "and no page reader could be loaded."
            ),
        }

    # ========================================================================
    # EXTRACT LINKS
    # ========================================================================

    async def extract_links(
        self,
        url: Optional[str] = None,
        limit: int = 200,
        **_: Any,
    ) -> Dict[str, Any]:

        if url is not None:

            if self._playwright_page is None:

                navigation = await self.navigate(
                    url
                )

                if not navigation.get(
                    "success"
                ):
                    return navigation

        if self._playwright_page is not None:

            try:

                links = await self._playwright_page.locator(
                    "a"
                ).evaluate_all(
                    """
                    elements => elements.map(
                        element => ({
                            text: (
                                element.innerText || ""
                            ).trim(),
                            href: element.href || ""
                        })
                    )
                    """
                )

                cleaned = []

                for link in links[:limit]:

                    href = str(
                        link.get(
                            "href",
                            ""
                        )
                    ).strip()

                    text = str(
                        link.get(
                            "text",
                            ""
                        )
                    ).strip()

                    if not href:
                        continue

                    cleaned.append(
                        {
                            "text": text,
                            "href": href,
                        }
                    )

                return {
                    "success": True,
                    "action": "extract_links",
                    "url": self._playwright_page.url,
                    "links": cleaned,
                    "count": len(cleaned),
                }

            except Exception as exc:

                return {
                    "success": False,
                    "error": str(exc),
                }

        return {
            "success": False,
            "error": (
                "No active browser page is available."
            ),
        }

    # ========================================================================
    # DOWNLOAD
    # ========================================================================

    async def download(
        self,
        url: str,
        destination: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        normalized_url = self._normalize_url(
            url
        )

        if self.download_manager is not None:

            for method_name in (
                "download",
                "download_file",
                "fetch",
            ):

                method = getattr(
                    self.download_manager,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            url=normalized_url,
                            destination=destination,
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "download",
                            "url": normalized_url,
                            "result": result,
                        }

                    except TypeError:

                        try:

                            result = method(
                                normalized_url,
                                destination,
                            )

                            if asyncio.iscoroutine(
                                result
                            ):
                                result = await result

                            return {
                                "success": True,
                                "action": "download",
                                "url": normalized_url,
                                "result": result,
                            }

                        except Exception:
                            pass

                    except Exception:
                        pass

        if (
            self._playwright_page is not None
        ):

            try:

                async with self._playwright_page.expect_download() as download_info:

                    await self._playwright_page.goto(
                        normalized_url
                    )

                download = await download_info.value

                target_directory = (
                    Path(destination).expanduser()
                    if destination
                    else self.download_root
                )

                target_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                target = (
                    target_directory
                    / download.suggested_filename
                )

                await download.save_as(
                    str(target)
                )

                return {
                    "success": True,
                    "action": "download",
                    "url": normalized_url,
                    "path": str(target),
                }

            except Exception as exc:

                return {
                    "success": False,
                    "error": str(exc),
                }

        return {
            "success": False,
            "error": (
                "No supported download manager or "
                "browser automation session is available."
            ),
        }

    # ========================================================================
    # BROWSER STATUS
    # ========================================================================

    async def browser_status(
        self,
        **_: Any,
    ) -> Dict[str, Any]:

        return {
            "success": True,
            "action": "browser_status",
            "playwright_available": (
                self._playwright is not None
            ),
            "automation_active": (
                self._playwright_browser is not None
            ),
            "page_active": (
                self._playwright_page is not None
            ),
            "current_url": (
                self._playwright_page.url
                if self._playwright_page is not None
                else None
            ),
            "browser_controller_available": (
                self.browser_controller is not None
            ),
            "search_engine_available": (
                self.web_search is not None
            ),
        }

    # ========================================================================
    # NEW TAB
    # ========================================================================

    async def new_tab(
        self,
        url: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        if self._playwright_browser is not None:

            try:

                page = await self._playwright_browser.new_page()

                self._playwright_page = page

                if url:

                    normalized_url = self._normalize_url(
                        url
                    )

                    response = await page.goto(
                        normalized_url,
                        wait_until="domcontentloaded",
                    )

                    return {
                        "success": True,
                        "action": "new_tab",
                        "url": page.url,
                        "status": (
                            response.status
                            if response is not None
                            else None
                        ),
                    }

                return {
                    "success": True,
                    "action": "new_tab",
                    "url": page.url,
                }

            except Exception as exc:

                return {
                    "success": False,
                    "error": str(exc),
                }

        if url:

            return await self.open_url(
                url,
                new_window=True,
            )

        try:

            opened = webbrowser.open(
                "about:blank",
                new=1,
            )

            return {
                "success": bool(opened),
                "action": "new_tab",
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # CLOSE BROWSER
    # ========================================================================

    async def close_browser(
        self,
        **_: Any,
    ) -> Dict[str, Any]:

        errors = []

        if self._playwright_page is not None:

            try:

                await self._playwright_page.close()

            except Exception as exc:

                errors.append(
                    str(exc)
                )

            finally:

                self._playwright_page = None

        if self._playwright_browser is not None:

            try:

                await self._playwright_browser.close()

            except Exception as exc:

                errors.append(
                    str(exc)
                )

            finally:

                self._playwright_browser = None

        if errors:

            return {
                "success": False,
                "action": "close_browser",
                "errors": errors,
            }

        return {
            "success": True,
            "action": "close_browser",
            "message": (
                "RENIX browser automation session closed."
            ),
        }

    # ========================================================================
    # FORM CONTROL
    # ========================================================================

    async def fill_form(
        self,
        fields: Dict[str, str],
        submit: bool = False,
        submit_selector: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        if not isinstance(
            fields,
            dict,
        ):

            return {
                "success": False,
                "error": (
                    "fields must be a dictionary."
                ),
            }

        if self.form_controller is not None:

            for method_name in (
                "fill",
                "fill_form",
                "populate",
            ):

                method = getattr(
                    self.form_controller,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            fields=fields,
                            submit=submit,
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "fill_form",
                            "result": result,
                        }

                    except Exception:
                        pass

        if self._playwright_page is None:

            return {
                "success": False,
                "error": (
                    "No active browser automation page "
                    "is available."
                ),
            }

        try:

            filled = []

            for selector, value in fields.items():

                locator = self._playwright_page.locator(
                    selector
                )

                await locator.fill(
                    str(value)
                )

                filled.append(
                    selector
                )

            if submit:

                if submit_selector:

                    await self._playwright_page.locator(
                        submit_selector
                    ).click()

                else:

                    await self._playwright_page.keyboard.press(
                        "Enter"
                    )

            return {
                "success": True,
                "action": "fill_form",
                "fields_filled": filled,
                "submitted": submit,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _truncate(
        value: Any,
        max_chars: int,
    ) -> Any:

        if not isinstance(
            value,
            str,
        ):
            return value

        if len(value) <= max_chars:
            return value

        return (
            value[:max_chars]
            + "\n\n[Content truncated by RENIX.]"
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def on_shutdown(self) -> None:
        """Close browser resources owned by this agent."""

        try:

            await self.close_browser()

        except Exception as exc:

            self.logger.debug(
                "Browser shutdown error: %s",
                exc,
            )


__all__ = [
    "BrowserAgent",
]


