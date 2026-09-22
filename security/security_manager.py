"""
RENIX Security Manager
======================

Central security controller for the RENIX AI system.

This module coordinates:

* Authentication
* Permissions
* Access control
* Confirmation handling
* Command safety
* Sandboxing
* Audit logging
* Threat monitoring
* Secure mode
* Emergency locking
* Secret management
  """

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("RENIX.SecurityManager")

class SecurityManager:
    """
    Central controller for all RENIX security systems.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the RENIX security manager.

        Args:
            config: Security configuration dictionary.
        """

        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.initialized = False

        self.components: Dict[str, Any] = {}

        if self.enabled:
            self._initialize_components()

        logger.info("SecurityManager created.")

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _initialize_components(self) -> None:
        """
        Initialize all security components.
        """

        self._initialize_authentication()
        self._initialize_permissions()
        self._initialize_access_control()
        self._initialize_confirmation()
        self._initialize_command_safety()
        self._initialize_sandbox()
        self._initialize_audit_logger()
        self._initialize_threat_monitor()
        self._initialize_secure_mode()
        self._initialize_emergency_lock()
        self._initialize_secrets_manager()

        self.initialized = True

        logger.info(
            "Security system initialized with %d components.",
            len(self.components),
        )

    # ============================================================
    # AUTHENTICATION
    # ============================================================

    def _initialize_authentication(self) -> None:
        """Initialize authentication manager."""

        try:
            from .authentication import AuthenticationManager

            self.components["authentication"] = AuthenticationManager(
                self.config.get("authentication", {})
            )

        except Exception as error:
            logger.warning(
                "Authentication initialization failed: %s",
                error,
            )

    # ============================================================
    # PERMISSIONS
    # ============================================================

    def _initialize_permissions(self) -> None:
        """Initialize permission manager."""

        try:
            from .permissions import PermissionManager

            self.components["permissions"] = PermissionManager(
                self.config.get("permissions", {})
            )

        except Exception as error:
            logger.warning(
                "Permission initialization failed: %s",
                error,
            )

    # ============================================================
    # ACCESS CONTROL
    # ============================================================

    def _initialize_access_control(self) -> None:
        """Initialize access controller."""

        try:
            from .access_control import AccessController

            self.components["access_control"] = AccessController(
                self.config.get("access_control", {})
            )

        except Exception as error:
            logger.warning(
                "Access control initialization failed: %s",
                error,
            )

    # ============================================================
    # CONFIRMATION
    # ============================================================

    def _initialize_confirmation(self) -> None:
        """Initialize confirmation manager."""

        try:
            from .confirmation import ConfirmationManager

            self.components["confirmation"] = ConfirmationManager(
                self.config.get("confirmation", {})
            )

        except Exception as error:
            logger.warning(
                "Confirmation initialization failed: %s",
                error,
            )

    # ============================================================
    # COMMAND SAFETY
    # ============================================================

    def _initialize_command_safety(self) -> None:
        """Initialize command safety system."""

        try:
            from .command_safety import CommandSafety

            self.components["command_safety"] = CommandSafety(
                self.config.get("command_safety", {})
            )

        except Exception as error:
            logger.warning(
                "Command safety initialization failed: %s",
                error,
            )

    # ============================================================
    # SANDBOX
    # ============================================================

    def _initialize_sandbox(self) -> None:
        """Initialize sandbox."""

        try:
            from .sandbox import Sandbox

            self.components["sandbox"] = Sandbox(
                self.config.get("sandbox", {})
            )

        except Exception as error:
            logger.warning(
                "Sandbox initialization failed: %s",
                error,
            )

    # ============================================================
    # AUDIT LOGGER
    # ============================================================

    def _initialize_audit_logger(self) -> None:
        """Initialize audit logger."""

        try:
            from .audit_logger import AuditLogger

            self.components["audit_logger"] = AuditLogger(
                self.config.get("audit_logger", {})
            )

        except Exception as error:
            logger.warning(
                "Audit logger initialization failed: %s",
                error,
            )

    # ============================================================
    # THREAT MONITOR
    # ============================================================

    def _initialize_threat_monitor(self) -> None:
        """Initialize threat monitor."""

        try:
            from .threat_monitor import ThreatMonitor

            self.components["threat_monitor"] = ThreatMonitor(
                self.config.get("threat_monitor", {})
            )

        except Exception as error:
            logger.warning(
                "Threat monitor initialization failed: %s",
                error,
            )

    # ============================================================
    # SECURE MODE
    # ============================================================

    def _initialize_secure_mode(self) -> None:
        """Initialize secure mode."""

        try:
            from .secure_mode import SecureMode

            self.components["secure_mode"] = SecureMode(
                self.config.get("secure_mode", {})
            )

        except Exception as error:
            logger.warning(
                "Secure mode initialization failed: %s",
                error,
            )

    # ============================================================
    # EMERGENCY LOCK
    # ============================================================

    def _initialize_emergency_lock(self) -> None:
        """Initialize emergency lock."""

        try:
            from .emergency_lock import EmergencyLock

            self.components["emergency_lock"] = EmergencyLock(
                self.config.get("emergency_lock", {})
            )

        except Exception as error:
            logger.warning(
                "Emergency lock initialization failed: %s",
                error,
            )

    # ============================================================
    # SECRETS MANAGER
    # ============================================================

    def _initialize_secrets_manager(self) -> None:
        """Initialize secrets manager."""

        try:
            from .secrets_manager import SecretsManager

            self.components["secrets_manager"] = SecretsManager(
                self.config.get("secrets_manager", {})
            )

        except Exception as error:
            logger.warning(
                "Secrets manager initialization failed: %s",
                error,
            )

    # ============================================================
    # COMPONENT ACCESS
    # ============================================================

    def get_component(
        self,
        name: str,
    ) -> Optional[Any]:
        """
        Get a registered security component.

        Args:
            name: Component name.

        Returns:
            Security component or None.
        """

        return self.components.get(name)

    # ============================================================
    # USER AUTHENTICATION
    # ============================================================

    def authenticate(
        self,
        credentials: Dict[str, Any],
    ) -> bool:
        """
        Authenticate a user.

        Args:
            credentials: Authentication credentials.

        Returns:
            True if authentication succeeds.
        """

        authentication = self.get_component(
            "authentication"
        )

        if authentication is None:
            logger.warning(
                "Authentication component unavailable."
            )
            return False

        try:
            authenticated = authentication.authenticate(
                credentials
            )

            self.log_event(
                event_type="authentication",
                details={
                    "success": bool(authenticated),
                },
            )

            return bool(authenticated)

        except Exception as error:
            logger.error(
                "Authentication error: %s",
                error,
            )

            return False

    # ============================================================
    # ACCESS CONTROL
    # ============================================================

    def check_access(
        self,
        user_id: str,
        resource: str,
        action: str,
    ) -> bool:
        """
        Check whether a user can access a resource.

        Args:
            user_id: User identifier.
            resource: Resource being accessed.
            action: Requested action.

        Returns:
            True if access is allowed.
        """

        access_control = self.get_component(
            "access_control"
        )

        if access_control is None:
            return False

        try:
            allowed = access_control.check_access(
                user_id=user_id,
                resource=resource,
                action=action,
            )

            self.log_event(
                event_type="access_check",
                user_id=user_id,
                details={
                    "resource": resource,
                    "action": action,
                    "allowed": allowed,
                },
            )

            return bool(allowed)

        except Exception as error:
            logger.error(
                "Access check failed: %s",
                error,
            )

            return False

    # ============================================================
    # PERMISSIONS
    # ============================================================

    def check_permission(
        self,
        user_id: str,
        command: str,
    ) -> bool:
        """
        Check whether a user has permission
        to execute a command.
        """

        permissions = self.get_component(
            "permissions"
        )

        if permissions is None:
            return False

        try:
            return bool(
                permissions.check_permission(
                    user_id=user_id,
                    command=command,
                )
            )

        except Exception as error:
            logger.error(
                "Permission check failed: %s",
                error,
            )

            return False

    # ============================================================
    # COMMAND SECURITY ANALYSIS
    # ============================================================

    def check_command(
        self,
        command: str,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Perform a complete security analysis
        before a command is executed.

        Returns:
            Dictionary containing:
            - allowed
            - requires_confirmation
            - risk_level
            - reason
        """

        result = {
            "allowed": True,
            "requires_confirmation": False,
            "risk_level": "low",
            "reason": "",
        }

        # Security disabled
        if not self.enabled:
            return result

        # Emergency lock
        if self.is_locked():

            result["allowed"] = False
            result["reason"] = (
                "RENIX security system is locked."
            )

            self.log_event(
                event_type="blocked_command",
                user_id=user_id,
                details={
                    "command": command,
                    "reason": result["reason"],
                },
            )

            return result

        # Secure mode
        secure_mode = self.get_component(
            "secure_mode"
        )

        if secure_mode is not None:

            try:
                if secure_mode.is_restricted(command):

                    result["allowed"] = False
                    result["reason"] = (
                        "Command restricted by secure mode."
                    )

                    return result

            except Exception as error:
                logger.error(
                    "Secure mode check failed: %s",
                    error,
                )

        # Command safety analysis
        command_safety = self.get_component(
            "command_safety"
        )

        if command_safety is not None:

            try:
                safety_result = command_safety.analyze(
                    command
                )

                if isinstance(safety_result, dict):
                    result.update(safety_result)

            except Exception as error:
                logger.error(
                    "Command safety analysis failed: %s",
                    error,
                )

        # Permission check
        if result.get("allowed", True):

            if not self.check_permission(
                user_id=user_id,
                command=command,
            ):

                result["allowed"] = False
                result["reason"] = (
                    "Permission denied."
                )

        # Confirmation check
        confirmation = self.get_component(
            "confirmation"
        )

        if (
            result.get("allowed", True)
            and confirmation is not None
        ):

            try:
                if confirmation.requires_confirmation(
                    command
                ):

                    result[
                        "requires_confirmation"
                    ] = True

            except Exception as error:
                logger.error(
                    "Confirmation check failed: %s",
                    error,
                )

        # Audit event
        self.log_event(
            event_type="command_check",
            user_id=user_id,
            details={
                "command": command,
                "result": result,
            },
        )

        return result

    # ============================================================
    # COMMAND AUTHORIZATION
    # ============================================================

    def authorize_command(
        self,
        command: str,
        user_id: str = "default",
    ) -> bool:
        """
        Check whether a command is authorized.
        """

        result = self.check_command(
            command=command,
            user_id=user_id,
        )

        return bool(
            result.get("allowed", False)
        )

    # ============================================================
    # CONFIRMATION CHECK
    # ============================================================

    def requires_confirmation(
        self,
        command: str,
        user_id: str = "default",
    ) -> bool:
        """
        Determine whether a command requires
        user confirmation.
        """

        result = self.check_command(
            command=command,
            user_id=user_id,
        )

        return bool(
            result.get(
                "requires_confirmation",
                False,
            )
        )

    # ============================================================
    # COMMAND EXECUTION APPROVAL
    # ============================================================

    def approve_command(
        self,
        command: str,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Determine whether RENIX may execute
        a command.

        This is the main security entry point.
        """

        result = self.check_command(
            command=command,
            user_id=user_id,
        )

        return {
            "approved": result.get(
                "allowed",
                False,
            ),
            "requires_confirmation": result.get(
                "requires_confirmation",
                False,
            ),
            "risk_level": result.get(
                "risk_level",
                "unknown",
            ),
            "reason": result.get(
                "reason",
                "",
            ),
        }

    # ============================================================
    # THREAT SCANNING
    # ============================================================

    def scan_for_threats(
        self,
        data: Any,
    ) -> Dict[str, Any]:
        """
        Scan input data for potential threats.
        """

        threat_monitor = self.get_component(
            "threat_monitor"
        )

        if threat_monitor is None:

            return {
                "threat_detected": False,
                "reason": (
                    "Threat monitor unavailable."
                ),
            }

        try:
            result = threat_monitor.scan(data)

            self.log_event(
                event_type="threat_scan",
                details={
                    "result": result,
                },
            )

            return result

        except Exception as error:

            logger.error(
                "Threat scan failed: %s",
                error,
            )

            return {
                "threat_detected": False,
                "reason": "Threat scan failed.",
            }

    # ============================================================
    # SANDBOX
    # ============================================================

    def get_sandbox(
        self,
    ) -> Optional[Any]:
        """
        Return the RENIX sandbox component.
        """

        return self.get_component(
            "sandbox"
        )

    # ============================================================
    # SECURE MODE
    # ============================================================

    def enable_secure_mode(
        self,
    ) -> bool:
        """
        Enable RENIX secure mode.
        """

        secure_mode = self.get_component(
            "secure_mode"
        )

        if secure_mode is None:
            return False

        try:

            secure_mode.enable()

            self.log_event(
                event_type="secure_mode_enabled"
            )

            logger.warning(
                "RENIX secure mode enabled."
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to enable secure mode: %s",
                error,
            )

            return False

    def disable_secure_mode(
        self,
    ) -> bool:
        """
        Disable RENIX secure mode.
        """

        secure_mode = self.get_component(
            "secure_mode"
        )

        if secure_mode is None:
            return False

        try:

            secure_mode.disable()

            self.log_event(
                event_type="secure_mode_disabled"
            )

            logger.info(
                "RENIX secure mode disabled."
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to disable secure mode: %s",
                error,
            )

            return False

    def is_secure_mode_enabled(
        self,
    ) -> bool:
        """
        Check whether secure mode is enabled.
        """

        secure_mode = self.get_component(
            "secure_mode"
        )

        if secure_mode is None:
            return False

        try:
            return bool(
                secure_mode.is_enabled()
            )

        except Exception:
            return False

    # ============================================================
    # EMERGENCY LOCK
    # ============================================================

    def lock(
        self,
        reason: str = "Manual security lock",
    ) -> bool:
        """
        Activate emergency lock.
        """

        emergency_lock = self.get_component(
            "emergency_lock"
        )

        if emergency_lock is None:
            return False

        try:

            emergency_lock.lock(
                reason=reason
            )

            self.log_event(
                event_type="security_lock",
                details={
                    "reason": reason,
                },
            )

            logger.warning(
                "RENIX security locked: %s",
                reason,
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to activate lock: %s",
                error,
            )

            return False

    def unlock(
        self,
        credentials: Optional[
            Dict[str, Any]
        ] = None,
    ) -> bool:
        """
        Attempt to unlock RENIX.
        """

        emergency_lock = self.get_component(
            "emergency_lock"
        )

        if emergency_lock is None:
            return False

        try:

            success = emergency_lock.unlock(
                credentials
            )

            self.log_event(
                event_type="security_unlock",
                details={
                    "success": bool(success),
                },
            )

            return bool(success)

        except Exception as error:

            logger.error(
                "Unlock failed: %s",
                error,
            )

            return False

    def is_locked(
        self,
    ) -> bool:
        """
        Check whether RENIX is locked.
        """

        emergency_lock = self.get_component(
            "emergency_lock"
        )

        if emergency_lock is None:
            return False

        try:
            return bool(
                emergency_lock.is_locked()
            )

        except Exception as error:

            logger.error(
                "Lock status check failed: %s",
                error,
            )

            return False

    # ============================================================
    # AUDIT LOGGING
    # ============================================================

    def log_event(
        self,
        event_type: str,
        user_id: str = "default",
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        """
        Log a security event.
        """

        audit_logger = self.get_component(
            "audit_logger"
        )

        if audit_logger is None:
            return

        try:

            audit_logger.log(
                event_type=event_type,
                user_id=user_id,
                details=details or {},
            )

        except Exception as error:

            logger.error(
                "Audit logging failed: %s",
                error,
            )

    # ============================================================
    # SECRETS
    # ============================================================

    def get_secret(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a secret securely.
        """

        secrets_manager = self.get_component(
            "secrets_manager"
        )

        if secrets_manager is None:
            return default

        try:
            return secrets_manager.get(
                key,
                default,
            )

        except Exception as error:

            logger.error(
                "Failed to retrieve secret: %s",
                error,
            )

            return default

    def set_secret(
        self,
        key: str,
        value: Any,
    ) -> bool:
        """
        Store a secret securely.
        """

        secrets_manager = self.get_component(
            "secrets_manager"
        )

        if secrets_manager is None:
            return False

        try:

            secrets_manager.set(
                key,
                value,
            )

            self.log_event(
                event_type="secret_updated",
                details={
                    "key": key,
                },
            )

            return True

        except Exception as error:

            logger.error(
                "Failed to store secret: %s",
                error,
            )

            return False

    # ============================================================
    # SECURITY STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return complete security system status.
        """

        component_status = {}

        for name, component in self.components.items():

            try:

                if hasattr(
                    component,
                    "get_status",
                ):

                    component_status[
                        name
                    ] = component.get_status()

                else:

                    component_status[
                        name
                    ] = {
                        "available": True
                    }

            except Exception:

                component_status[name] = {
                    "available": False
                }

        return {
            "enabled": self.enabled,
            "initialized": self.initialized,
            "locked": self.is_locked(),
            "secure_mode": (
                self.is_secure_mode_enabled()
            ),
            "component_count": len(
                self.components
            ),
            "components": component_status,
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shut down all security components.
        """

        logger.info(
            "Shutting down RENIX security system..."
        )

        for name, component in list(
            self.components.items()
        ):

            try:

                shutdown_method = getattr(
                    component,
                    "shutdown",
                    None,
                )

                if callable(
                    shutdown_method
                ):

                    shutdown_method()

                logger.info(
                    "Security component stopped: %s",
                    name,
                )

            except Exception as error:

                logger.error(
                    "Failed to stop security component %s: %s",
                    name,
                    error,
                )

        self.components.clear()
        self.initialized = False

        logger.info(
            "RENIX security system shutdown complete."
        )
