"""
RENIX Browser Security
======================

Security layer for browser operations.

Responsibilities:
    - Validate URLs
    - Block dangerous schemes
    - Detect suspicious navigation
    - Restrict downloads
    - Prevent local-file access when disabled
    - Control allowed domains
    - Detect basic phishing indicators
    - Provide safe navigation checks
"""

from __future__ import annotations

import ipaddress
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class SecurityResult:
    """Result of a browser security inspection."""

    allowed: bool
    url: str
    reason: str = ""
    warnings: list[str] = field(
        default_factory=list
    )
    risk_level: str = "low"

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "url": self.url,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "risk_level": self.risk_level,
        }


class BrowserSecurity:
    """Security policy engine for RENIX browser operations."""

    SAFE_SCHEMES = {
        "http",
        "https",
    }

    BLOCKED_SCHEMES = {
        "file",
        "javascript",
        "vbscript",
        "data",
        "blob",
        "chrome",
        "chrome-extension",
        "devtools",
        "about",
    }

    SUSPICIOUS_PORTS = {
        21,
        22,
        23,
        25,
        110,
        135,
        139,
        143,
        445,
        3389,
    }

    PRIVATE_HOSTNAMES = {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }

    def __init__(
        self,
        *,
        allow_http: bool = True,
        allow_https: bool = True,
        allow_local_network: bool = False,
        allow_private_ips: bool = False,
        allowed_domains: Iterable[str] | None = None,
        blocked_domains: Iterable[str] | None = None,
        max_url_length: int = 8192,
    ) -> None:

        self.allow_http = allow_http
        self.allow_https = allow_https
        self.allow_local_network = allow_local_network
        self.allow_private_ips = allow_private_ips
        self.max_url_length = max_url_length

        self.allowed_domains = {
            self._normalize_domain(domain)
            for domain in (
                allowed_domains or []
            )
            if domain
        }

        self.blocked_domains = {
            self._normalize_domain(domain)
            for domain in (
                blocked_domains or []
            )
            if domain
        }

    # ========================================================
    # URL VALIDATION
    # ========================================================

    def validate_url(
        self,
        url: str,
        *,
        require_https: bool = False,
    ) -> SecurityResult:
        """Validate a URL against the browser security policy."""

        url = str(
            url or ""
        ).strip()

        if not url:
            return SecurityResult(
                allowed=False,
                url=url,
                reason="URL is empty.",
                risk_level="high",
            )

        if len(url) > self.max_url_length:

            return SecurityResult(
                allowed=False,
                url=url,
                reason="URL exceeds the maximum allowed length.",
                risk_level="high",
            )

        parsed = urlparse(
            url
        )

        scheme = (
            parsed.scheme.lower()
        )

        if scheme in self.BLOCKED_SCHEMES:

            return SecurityResult(
                allowed=False,
                url=url,
                reason=f"Blocked URL scheme: {scheme}",
                risk_level="high",
            )

        if scheme not in self.SAFE_SCHEMES:

            return SecurityResult(
                allowed=False,
                url=url,
                reason=f"Unsupported URL scheme: {scheme or 'none'}",
                risk_level="high",
            )

        if scheme == "http" and not self.allow_http:

            return SecurityResult(
                allowed=False,
                url=url,
                reason="HTTP navigation is disabled.",
                risk_level="medium",
            )

        if scheme == "https" and not self.allow_https:

            return SecurityResult(
                allowed=False,
                url=url,
                reason="HTTPS navigation is disabled.",
                risk_level="high",
            )

        if require_https and scheme != "https":

            return SecurityResult(
                allowed=False,
                url=url,
                reason="HTTPS is required.",
                risk_level="high",
            )

        if not parsed.netloc:

            return SecurityResult(
                allowed=False,
                url=url,
                reason="URL does not contain a valid host.",
                risk_level="high",
            )

        hostname = (
            parsed.hostname or ""
        ).lower()

        if not hostname:

            return SecurityResult(
                allowed=False,
                url=url,
                reason="URL hostname is missing.",
                risk_level="high",
            )

        return self._inspect_host(
            url,
            hostname,
            parsed,
        )

    # ========================================================
    # HOST INSPECTION
    # ========================================================

    def _inspect_host(
        self,
        url: str,
        hostname: str,
        parsed,
    ) -> SecurityResult:

        warnings: list[str] = []

        normalized = self._normalize_domain(
            hostname
        )

        # Explicitly blocked domains.
        if self._matches_domain(
            normalized,
            self.blocked_domains,
        ):

            return SecurityResult(
                allowed=False,
                url=url,
                reason="Domain is blocked by browser security policy.",
                warnings=warnings,
                risk_level="high",
            )

        # Allowlist enforcement.
        if self.allowed_domains:

            if not self._matches_domain(
                normalized,
                self.allowed_domains,
            ):

                return SecurityResult(
                    allowed=False,
                    url=url,
                    reason="Domain is not present in the browser allowlist.",
                    warnings=warnings,
                    risk_level="medium",
                )

        # Local hostname.
        if hostname in self.PRIVATE_HOSTNAMES:

            if not self.allow_local_network:

                return SecurityResult(
                    allowed=False,
                    url=url,
                    reason="Localhost access is disabled.",
                    warnings=warnings,
                    risk_level="high",
                )

            warnings.append(
                "Target is localhost."
            )

        # IP address inspection.
        ip_info = self._inspect_ip(
            hostname
        )

        if ip_info is not None:

            is_private = ip_info[
                "private"
            ]

            is_loopback = ip_info[
                "loopback"
            ]

            is_link_local = ip_info[
                "link_local"
            ]

            if (
                is_private
                or is_loopback
                or is_link_local
            ):

                if not self.allow_private_ips:

                    return SecurityResult(
                        allowed=False,
                        url=url,
                        reason=(
                            "Navigation to a private or "
                            "local IP address is disabled."
                        ),
                        warnings=warnings,
                        risk_level="high",
                    )

                warnings.append(
                    "Target is a private/local IP address."
                )

        # Suspicious port.
        port = parsed.port

        if port in self.SUSPICIOUS_PORTS:

            warnings.append(
                f"Suspicious network port detected: {port}"
            )

        # Userinfo in URL can hide the actual destination.
        if parsed.username or parsed.password:

            warnings.append(
                "URL contains user information before the hostname."
            )

        # Punycode / internationalized domain warning.
        if hostname.startswith(
            "xn--"
        ) or ".xn--" in hostname:

            warnings.append(
                "Internationalized/punycode domain detected."
            )

        # Excessive subdomains can be suspicious.
        if hostname.count(".") >= 5:

            warnings.append(
                "URL contains an unusually deep subdomain chain."
            )

        # Numeric-only hostname can sometimes be suspicious.
        if self._looks_like_encoded_ip(
            hostname
        ):

            warnings.append(
                "Hostname may represent an encoded IP address."
            )

        risk = (
            "medium"
            if warnings
            else "low"
        )

        return SecurityResult(
            allowed=True,
            url=url,
            reason="URL passed browser security checks.",
            warnings=warnings,
            risk_level=risk,
        )

    # ========================================================
    # NAVIGATION
    # ========================================================

    def can_navigate(
        self,
        url: str,
        *,
        require_https: bool = False,
    ) -> bool:

        result = self.validate_url(
            url,
            require_https=require_https,
        )

        return result.allowed

    def check_navigation(
        self,
        url: str,
        *,
        require_https: bool = False,
    ) -> SecurityResult:

        return self.validate_url(
            url,
            require_https=require_https,
        )

    # ========================================================
    # REDIRECT VALIDATION
    # ========================================================

    def validate_redirect(
        self,
        original_url: str,
        redirect_url: str,
    ) -> SecurityResult:
        """
        Validate a redirect destination.

        The redirect is checked independently rather than
        trusting the original destination.
        """

        result = self.validate_url(
            redirect_url
        )

        if not result.allowed:
            return result

        original = urlparse(
            original_url
        )

        target = urlparse(
            redirect_url
        )

        original_host = (
            original.hostname or ""
        ).lower()

        target_host = (
            target.hostname or ""
        ).lower()

        if (
            original_host
            and target_host
            and original_host != target_host
        ):

            result.warnings.append(
                "Redirect changes the destination domain."
            )

            if result.risk_level == "low":
                result.risk_level = "medium"

        return result

    # ========================================================
    # DOWNLOAD VALIDATION
    # ========================================================

    def validate_download_url(
        self,
        url: str,
    ) -> SecurityResult:

        return self.validate_url(
            url
        )

    def validate_download_filename(
        self,
        filename: str,
    ) -> SecurityResult:
        """Validate a proposed local download filename."""

        filename = str(
            filename or ""
        ).strip()

        if not filename:

            return SecurityResult(
                allowed=False,
                url=filename,
                reason="Filename is empty.",
                risk_level="high",
            )

        if len(filename) > 255:

            return SecurityResult(
                allowed=False,
                url=filename,
                reason="Filename is too long.",
                risk_level="medium",
            )

        dangerous_patterns = (
            "..",
            "/",
            "\\",
            "\x00",
        )

        for pattern in dangerous_patterns:

            if pattern in filename:

                return SecurityResult(
                    allowed=False,
                    url=filename,
                    reason=(
                        "Filename contains a potentially "
                        "unsafe path sequence."
                    ),
                    risk_level="high",
                )

        return SecurityResult(
            allowed=True,
            url=filename,
            reason="Filename passed security checks.",
            risk_level="low",
        )

    # ========================================================
    # DOMAIN MANAGEMENT
    # ========================================================

    def add_allowed_domain(
        self,
        domain: str,
    ) -> None:

        normalized = self._normalize_domain(
            domain
        )

        if normalized:
            self.allowed_domains.add(
                normalized
            )

    def remove_allowed_domain(
        self,
        domain: str,
    ) -> None:

        normalized = self._normalize_domain(
            domain
        )

        self.allowed_domains.discard(
            normalized
        )

    def add_blocked_domain(
        self,
        domain: str,
    ) -> None:

        normalized = self._normalize_domain(
            domain
        )

        if normalized:
            self.blocked_domains.add(
                normalized
            )

    def remove_blocked_domain(
        self,
        domain: str,
    ) -> None:

        normalized = self._normalize_domain(
            domain
        )

        self.blocked_domains.discard(
            normalized
        )

    # ========================================================
    # DOMAIN CHECKS
    # ========================================================

    def is_domain_allowed(
        self,
        domain: str,
    ) -> bool:

        normalized = self._normalize_domain(
            domain
        )

        if self._matches_domain(
            normalized,
            self.blocked_domains,
        ):
            return False

        if not self.allowed_domains:
            return True

        return self._matches_domain(
            normalized,
            self.allowed_domains,
        )

    def is_domain_blocked(
        self,
        domain: str,
    ) -> bool:

        normalized = self._normalize_domain(
            domain
        )

        return self._matches_domain(
            normalized,
            self.blocked_domains,
        )

    # ========================================================
    # PHISHING INDICATORS
    # ========================================================

    def phishing_indicators(
        self,
        url: str,
    ) -> list[str]:
        """
        Return basic heuristic phishing indicators.

        This is not a replacement for a real threat-
        intelligence or anti-phishing service.
        """

        indicators: list[str] = []

        parsed = urlparse(
            url
        )

        hostname = (
            parsed.hostname or ""
        ).lower()

        if not hostname:
            return [
                "Missing hostname."
            ]

        # IP instead of domain.
        if self._inspect_ip(
            hostname
        ) is not None:

            indicators.append(
                "Destination uses an IP address instead of a domain."
            )

        # Punycode.
        if (
            hostname.startswith("xn--")
            or ".xn--" in hostname
        ):

            indicators.append(
                "Punycode domain detected."
            )

        # Excessive hyphens.
        if hostname.count("-") >= 4:

            indicators.append(
                "Domain contains many hyphens."
            )

        # Excessive subdomains.
        if hostname.count(".") >= 5:

            indicators.append(
                "Domain contains many subdomains."
            )

        # URL userinfo.
        if parsed.username or parsed.password:

            indicators.append(
                "URL contains userinfo before the hostname."
            )

        # Suspicious keywords.
        suspicious_words = (
            "login",
            "verify",
            "verification",
            "secure",
            "account",
            "password",
            "wallet",
            "update",
            "confirm",
        )

        combined = (
            url.lower()
        )

        matched = [
            word
            for word in suspicious_words
            if word in combined
        ]

        if len(matched) >= 3:

            indicators.append(
                "URL contains multiple sensitive-account keywords."
            )

        return indicators

    # ========================================================
    # SAFE URL
    # ========================================================

    def sanitize_url(
        self,
        url: str,
    ) -> str:

        url = str(
            url or ""
        ).strip()

        parsed = urlparse(
            url
        )

        if parsed.scheme.lower() not in {
            "http",
            "https",
        }:

            return ""

        if not parsed.hostname:

            return ""

        # Reconstruct without credentials.
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname

        port = ""

        if parsed.port is not None:

            default_port = (
                443
                if scheme == "https"
                else 80
            )

            if parsed.port != default_port:
                port = f":{parsed.port}"

        return (
            f"{scheme}://"
            f"{hostname}"
            f"{port}"
            f"{parsed.path}"
            f"{('?' + parsed.query) if parsed.query else ''}"
            f"{('#' + parsed.fragment) if parsed.fragment else ''}"
        )

    # ========================================================
    # PRIVATE IP
    # ========================================================

    @staticmethod
    def _inspect_ip(
        hostname: str,
    ) -> dict | None:

        try:

            ip = ipaddress.ip_address(
                hostname
            )

        except ValueError:
            return None

        return {
            "private": ip.is_private,
            "loopback": ip.is_loopback,
            "link_local": ip.is_link_local,
            "multicast": ip.is_multicast,
            "reserved": ip.is_reserved,
        }

    # ========================================================
    # ENCODED IP
    # ========================================================

    @staticmethod
    def _looks_like_encoded_ip(
        hostname: str,
    ) -> bool:

        # Decimal integer IP representation.
        if hostname.isdigit():

            try:
                value = int(
                    hostname
                )

                return (
                    0
                    <= value
                    <= 4294967295
                )

            except ValueError:
                return False

        # Hexadecimal-looking hostname.
        if re.fullmatch(
            r"0x[0-9a-fA-F]+",
            hostname,
        ):

            return True

        return False

    # ========================================================
    # DOMAIN HELPERS
    # ========================================================

    @staticmethod
    def _normalize_domain(
        domain: str,
    ) -> str:

        domain = str(
            domain or ""
        ).strip().lower()

        domain = domain.removeprefix(
            "http://"
        )

        domain = domain.removeprefix(
            "https://"
        )

        domain = domain.split(
            "/",
            1,
        )[0]

        domain = domain.lstrip(
            "."
        )

        return domain.rstrip(
            "."
        )

    @staticmethod
    def _matches_domain(
        hostname: str,
        domains: set[str],
    ) -> bool:

        hostname = hostname.lower().rstrip(
            "."
        )

        for domain in domains:

            domain = domain.lower().rstrip(
                "."
            )

            if hostname == domain:
                return True

            if hostname.endswith(
                "." + domain
            ):
                return True

        return False


__all__ = [
    "BrowserSecurity",
    "SecurityResult",
]


