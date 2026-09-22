"""
RENIX Browser Link Handler
==========================

Handles links on browser pages.

Responsibilities:
    - Discover links
    - Read link information
    - Open links
    - Follow links
    - Open links in a new page when supported
    - Resolve relative URLs
    - Filter internal/external links
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class LinkHandler:
    """Manage links on browser pages."""

    def __init__(
        self,
        browser_manager: Any = None,
    ) -> None:
        self.browser_manager = browser_manager

    # ========================================================
    # PAGE
    # ========================================================

    def _get_page(
        self,
        page_id: str | None = None,
    ) -> Any:

        if self.browser_manager is None:
            raise RuntimeError(
                "BrowserManager is not configured."
            )

        return self.browser_manager.get_page(
            page_id
        )

    # ========================================================
    # DISCOVER LINKS
    # ========================================================

    def get_links(
        self,
        *,
        page_id: str | None = None,
        include_empty: bool = False,
    ) -> list[dict[str, Any]]:
        """Return all links from the current page."""

        page = self._get_page(
            page_id
        )

        script = """
        () => Array.from(
            document.querySelectorAll("a")
        ).map((anchor, index) => ({
            index,
            text:
                (anchor.innerText || anchor.textContent || "")
                .trim(),
            href:
                anchor.href || "",
            raw_href:
                anchor.getAttribute("href") || "",
            title:
                anchor.getAttribute("title") || "",
            target:
                anchor.getAttribute("target") || "",
            rel:
                anchor.getAttribute("rel") || "",
            download:
                anchor.hasAttribute("download"),
            aria_label:
                anchor.getAttribute("aria-label") || ""
        }))
        """

        links = self._evaluate(
            page,
            script,
        )

        if include_empty:
            return links

        return [
            link
            for link in links
            if link.get("href")
        ]

    # ========================================================
    # FIND LINK
    # ========================================================

    def find_link(
        self,
        text: str | None = None,
        *,
        href: str | None = None,
        contains: bool = True,
        page_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Find the first matching link."""

        links = self.get_links(
            page_id=page_id
        )

        target_text = (
            text.strip().lower()
            if text
            else None
        )

        target_href = (
            href.strip()
            if href
            else None
        )

        for link in links:

            link_text = str(
                link.get("text", "")
            ).strip().lower()

            link_href = str(
                link.get("href", "")
            ).strip()

            text_matches = True
            href_matches = True

            if target_text is not None:

                if contains:
                    text_matches = (
                        target_text
                        in link_text
                    )
                else:
                    text_matches = (
                        target_text
                        == link_text
                    )

            if target_href is not None:

                href_matches = (
                    link_href
                    == target_href
                    or target_href
                    in link_href
                    if contains
                    else link_href
                    == target_href
                )

            if text_matches and href_matches:
                return link

        return None

    # ========================================================
    # CLICK
    # ========================================================

    def click(
        self,
        selector: str,
        *,
        page_id: str | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Click a link using a CSS selector."""

        page = self._get_page(
            page_id
        )

        locator = self._locator(
            page,
            selector,
        )

        click = getattr(
            locator,
            "click",
            None,
        )

        if not callable(click):
            raise RuntimeError(
                "Browser backend does not support link clicking."
            )

        if timeout is not None:

            try:
                click(
                    timeout=timeout * 1000
                )
            except TypeError:
                click()

        else:
            click()

        return True

    # ========================================================
    # CLICK BY TEXT
    # ========================================================

    def click_text(
        self,
        text: str,
        *,
        exact: bool = False,
        page_id: str | None = None,
    ) -> bool:
        """Click a link based on visible text."""

        page = self._get_page(
            page_id
        )

        get_by_text = getattr(
            page,
            "get_by_text",
            None,
        )

        if callable(get_by_text):

            locator = get_by_text(
                text,
                exact=exact,
            )

            click = getattr(
                locator,
                "click",
                None,
            )

            if callable(click):
                click()
                return True

        links = self.get_links(
            page_id=page_id
        )

        target = text.strip().lower()

        for link in links:

            link_text = str(
                link.get("text", "")
            ).strip().lower()

            matches = (
                link_text == target
                if exact
                else target in link_text
            )

            if matches:

                href = link.get(
                    "href"
                )

                if href:
                    self.open(
                        href,
                        page_id=page_id,
                    )

                    return True

        return False

    # ========================================================
    # OPEN
    # ========================================================

    def open(
        self,
        url: str,
        *,
        page_id: str | None = None,
        wait_until: str | None = None,
    ) -> Any:
        """Navigate the current page to a URL."""

        page = self._get_page(
            page_id
        )

        if not url or not url.strip():
            raise ValueError(
                "URL cannot be empty."
            )

        url = url.strip()

        goto = getattr(
            page,
            "goto",
            None,
        )

        if not callable(goto):
            raise RuntimeError(
                "Browser backend does not support navigation."
            )

        if wait_until is not None:

            try:
                return goto(
                    url,
                    wait_until=wait_until,
                )
            except TypeError:
                return goto(
                    url
                )

        return goto(
            url
        )

    # ========================================================
    # OPEN LINK
    # ========================================================

    def open_link(
        self,
        link: str | dict[str, Any],
        *,
        page_id: str | None = None,
    ) -> Any:
        """Open a link represented by a URL or link dictionary."""

        if isinstance(
            link,
            dict,
        ):
            url = link.get(
                "href"
            )

        else:
            url = link

        if not url:
            raise ValueError(
                "Link does not contain a URL."
            )

        return self.open(
            str(url),
            page_id=page_id,
        )

    # ========================================================
    # OPEN IN NEW PAGE
    # ========================================================

    def open_new_page(
        self,
        url: str,
        *,
        page_id: str | None = None,
    ) -> Any:
        """
        Open URL in a new browser page/tab when supported.
        """

        if self.browser_manager is None:
            raise RuntimeError(
                "BrowserManager is not configured."
            )

        new_page = getattr(
            self.browser_manager,
            "new_page",
            None,
        )

        if callable(new_page):

            page = new_page()

            goto = getattr(
                page,
                "goto",
                None,
            )

            if callable(goto):
                goto(
                    url
                )

            return page

        # Fallback: use target=_blank through JS.
        page = self._get_page(
            page_id
        )

        script = """
        (url) => {
            window.open(
                url,
                "_blank",
                "noopener,noreferrer"
            );

            return true;
        }
        """

        return self._evaluate(
            page,
            script,
            url,
        )

    # ========================================================
    # FOLLOW
    # ========================================================

    def follow(
        self,
        *,
        text: str | None = None,
        href: str | None = None,
        selector: str | None = None,
        page_id: str | None = None,
    ) -> bool:
        """Find and follow a link."""

        if selector:

            return self.click(
                selector,
                page_id=page_id,
            )

        link = self.find_link(
            text=text,
            href=href,
            page_id=page_id,
        )

        if not link:
            return False

        url = link.get(
            "href"
        )

        if not url:
            return False

        self.open(
            url,
            page_id=page_id,
        )

        return True

    # ========================================================
    # RESOLVE URL
    # ========================================================

    @staticmethod
    def resolve_url(
        href: str,
        base_url: str,
    ) -> str:

        return urljoin(
            base_url,
            href,
        )

    # ========================================================
    # URL TYPE
    # ========================================================

    @staticmethod
    def is_absolute(
        url: str,
    ) -> bool:

        parsed = urlparse(
            url
        )

        return bool(
            parsed.scheme
            and parsed.netloc
        )

    @staticmethod
    def is_external(
        url: str,
        base_url: str,
    ) -> bool:

        try:

            target = urlparse(
                url
            )

            base = urlparse(
                base_url
            )

            if not target.netloc:
                return False

            return (
                target.netloc.lower()
                != base.netloc.lower()
            )

        except Exception:

            return False

    # ========================================================
    # FILTER LINKS
    # ========================================================

    def get_internal_links(
        self,
        *,
        page_id: str | None = None,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:

        links = self.get_links(
            page_id=page_id
        )

        if not base_url:
            return links

        return [
            link
            for link in links
            if not self.is_external(
                link.get("href", ""),
                base_url,
            )
        ]

    def get_external_links(
        self,
        *,
        page_id: str | None = None,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:

        if not base_url:
            return []

        links = self.get_links(
            page_id=page_id
        )

        return [
            link
            for link in links
            if self.is_external(
                link.get("href", ""),
                base_url,
            )
        ]

    # ========================================================
    # DOWNLOAD LINKS
    # ========================================================

    def get_download_links(
        self,
        *,
        page_id: str | None = None,
    ) -> list[dict[str, Any]]:

        links = self.get_links(
            page_id=page_id
        )

        return [
            link
            for link in links
            if link.get("download")
        ]

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _locator(
        page: Any,
        selector: str,
    ) -> Any:

        locator = getattr(
            page,
            "locator",
            None,
        )

        if not callable(locator):
            raise RuntimeError(
                "Browser page does not support locators."
            )

        return locator(
            selector
        )

    @staticmethod
    def _evaluate(
        page: Any,
        script: str,
        argument: Any = None,
    ) -> Any:

        evaluator = getattr(
            page,
            "evaluate",
            None,
        )

        if not callable(evaluator):
            raise RuntimeError(
                "Browser page does not support evaluation."
            )

        if argument is None:
            return evaluator(
                script
            )

        try:
            return evaluator(
                script,
                argument,
            )
        except TypeError:
            return evaluator(
                f"""
                () => {{
                    const args = {argument!r};
                    return ({script})(args);
                }}
                """
            )


__all__ = [
    "LinkHandler",
]


