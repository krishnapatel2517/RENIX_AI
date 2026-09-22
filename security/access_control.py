"""
RENIX Access Control
====================

Controls access to RENIX capabilities and resources.

Features:

* Permission-based access checking
* Role-based access control integration
* Resource protection
* Action protection
* Access policies
* Access rules
* Temporary access restrictions
* Context-aware access checks
* Security decision logging
  """

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from security.permissions import PermissionsManager

logger = logging.getLogger("RENIX.AccessControl")

class AccessControl:
    """
    Central access control layer for RENIX.

    This class decides whether a user is allowed
    to access a resource or perform an action.

    Example:

        access_control.check_access(
            user_id="krishna",
            resource="files",
            action="delete",
        )
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        permissions: Optional[
            PermissionsManager
        ] = None,
    ) -> None:
        """
        Initialize RENIX access control.
        """

        self.config = config or {}

        self.enabled = self.config.get(
            "enabled",
            True,
        )

        self.default_allow = self.config.get(
            "default_allow",
            False,
        )

        self.permissions = (
            permissions
            or PermissionsManager()
        )

        self.policies: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.rules: List[
            Dict[str, Any]
        ] = []

        self.access_history: List[
            Dict[str, Any]
        ] = []

        self.max_history = int(
            self.config.get(
                "max_history",
                1000,
            )
        )

        logger.info(
            "AccessControl initialized."
        )

    # ============================================================
    # POLICY MANAGEMENT
    # ============================================================

    def register_policy(
        self,
        resource: str,
        required_permission: str,
        description: str = "",
    ) -> bool:
        """
        Register an access policy.

        Example:

            register_policy(
                resource="files.delete",
                required_permission="files.delete"
            )
        """

        if not resource:
            return False

        self.policies[resource] = {
            "permission": (
                required_permission
            ),
            "description": description,
            "created_at": (
                datetime.now().isoformat()
            ),
            "enabled": True,
        }

        logger.info(
            "Access policy registered: %s",
            resource,
        )

        return True

    def remove_policy(
        self,
        resource: str,
    ) -> bool:
        """
        Remove an access policy.
        """

        if resource not in self.policies:
            return False

        del self.policies[resource]

        logger.info(
            "Access policy removed: %s",
            resource,
        )

        return True

    def enable_policy(
        self,
        resource: str,
    ) -> bool:
        """
        Enable a policy.
        """

        policy = self.policies.get(
            resource
        )

        if policy is None:
            return False

        policy["enabled"] = True

        return True

    def disable_policy(
        self,
        resource: str,
    ) -> bool:
        """
        Disable a policy.
        """

        policy = self.policies.get(
            resource
        )

        if policy is None:
            return False

        policy["enabled"] = False

        return True

    # ============================================================
    # RULE MANAGEMENT
    # ============================================================

    def add_rule(
        self,
        name: str,
        condition: Dict[str, Any],
        effect: str = "deny",
        priority: int = 100,
    ) -> bool:
        """
        Add a custom access rule.

        Effects:
            allow
            deny

        Example:

            add_rule(
                name="block_guest_delete",
                condition={
                    "role": "guest",
                    "permission": "files.delete"
                },
                effect="deny"
            )
        """

        if effect not in (
            "allow",
            "deny",
        ):
            return False

        rule = {
            "name": name,
            "condition": condition,
            "effect": effect,
            "priority": priority,
            "enabled": True,
            "created_at": (
                datetime.now().isoformat()
            ),
        }

        self.rules.append(
            rule
        )

        self.rules.sort(
            key=lambda item: item[
                "priority"
            ]
        )

        logger.info(
            "Access rule added: %s",
            name,
        )

        return True

    def remove_rule(
        self,
        name: str,
    ) -> bool:
        """
        Remove an access rule.
        """

        for rule in self.rules:

            if rule["name"] == name:

                self.rules.remove(
                    rule
                )

                return True

        return False

    def enable_rule(
        self,
        name: str,
    ) -> bool:
        """
        Enable a rule.
        """

        for rule in self.rules:

            if rule["name"] == name:

                rule["enabled"] = True

                return True

        return False

    def disable_rule(
        self,
        name: str,
    ) -> bool:
        """
        Disable a rule.
        """

        for rule in self.rules:

            if rule["name"] == name:

                rule["enabled"] = False

                return True

        return False

    # ============================================================
    # RESOURCE PERMISSION
    # ============================================================

    @staticmethod
    def build_permission(
        resource: str,
        action: str,
    ) -> str:
        """
        Build permission string.

        Example:

            resource="files"
            action="delete"

        Result:

            files.delete
        """

        resource = resource.strip(
            "."
        )

        action = action.strip(
            "."
        )

        return (
            f"{resource}.{action}"
        )

    # ============================================================
    # POLICY LOOKUP
    # ============================================================

    def get_required_permission(
        self,
        resource: str,
        action: Optional[
            str
        ] = None,
    ) -> Optional[str]:
        """
        Determine required permission.
        """

        if action:

            permission = (
                self.build_permission(
                    resource,
                    action,
                )
            )

        else:

            permission = resource

        # Exact policy.
        policy = self.policies.get(
            permission
        )

        if (
            policy
            and policy.get(
                "enabled",
                False,
            )
        ):

            return policy.get(
                "permission"
            )

        # Wildcard policy search.
        parts = permission.split(
            "."
        )

        if len(parts) > 1:

            wildcard = (
                f"{parts[0]}.*"
            )

            policy = self.policies.get(
                wildcard
            )

            if (
                policy
                and policy.get(
                    "enabled",
                    False,
                )
            ):

                return policy.get(
                    "permission"
                )

        # Default behavior:
        # resource.action itself is permission.
        return permission

    # ============================================================
    # RULE MATCHING
    # ============================================================

    def _rule_matches(
        self,
        rule: Dict[str, Any],
        user_id: str,
        permission: str,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> bool:
        """
        Check whether a rule matches
        the current access request.
        """

        if not rule.get(
            "enabled",
            False,
        ):

            return False

        condition = rule.get(
            "condition",
            {}
        )

        # Match user ID.
        rule_user = condition.get(
            "user_id"
        )

        if (
            rule_user is not None
            and rule_user != user_id
        ):

            return False

        # Match permission.
        rule_permission = condition.get(
            "permission"
        )

        if (
            rule_permission is not None
            and rule_permission != permission
        ):

            return False

        # Match role.
        rule_role = condition.get(
            "role"
        )

        if rule_role is not None:

            user_roles = (
                self.permissions.get_user_roles(
                    user_id
                )
            )

            if rule_role not in user_roles:
                return False

        # Match context.
        rule_context = condition.get(
            "context"
        )

        if rule_context:

            context = context or {}

            for key, value in (
                rule_context.items()
            ):

                if context.get(
                    key
                ) != value:

                    return False

        return True

    # ============================================================
    # EVALUATE RULES
    # ============================================================

    def _evaluate_rules(
        self,
        user_id: str,
        permission: str,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[bool]:
        """
        Evaluate access rules.

        Returns:
            True  -> explicitly allowed
            False -> explicitly denied
            None  -> no matching rule
        """

        for rule in self.rules:

            if self._rule_matches(
                rule,
                user_id,
                permission,
                context,
            ):

                effect = rule.get(
                    "effect"
                )

                if effect == "allow":
                    return True

                if effect == "deny":
                    return False

        return None

    # ============================================================
    # ACCESS CHECK
    # ============================================================

    def check_access(
        self,
        user_id: str,
        resource: str,
        action: Optional[
            str
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Check whether a user can access
        a resource.

        Returns a detailed access decision.
        """

        permission = (
            self.get_required_permission(
                resource,
                action,
            )
        )

        result = {
            "allowed": False,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "permission": permission,
            "reason": "",
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        # If access control is disabled.
        if not self.enabled:

            result["allowed"] = True

            result["reason"] = (
                "Access control disabled."
            )

            self._record_access(
                result
            )

            return result

        # Evaluate custom rules first.
        rule_decision = (
            self._evaluate_rules(
                user_id,
                permission,
                context,
            )
        )

        if rule_decision is False:

            result["allowed"] = False

            result["reason"] = (
                "Denied by access rule."
            )

            self._record_access(
                result
            )

            logger.warning(
                "Access denied by rule: %s -> %s",
                user_id,
                permission,
            )

            return result

        if rule_decision is True:

            result["allowed"] = True

            result["reason"] = (
                "Allowed by access rule."
            )

            self._record_access(
                result
            )

            return result

        # Check permissions.
        if permission:

            allowed = (
                self.permissions.has_permission(
                    user_id,
                    permission,
                )
            )

            result["allowed"] = allowed

            if allowed:

                result["reason"] = (
                    "Permission granted."
                )

            else:

                result["reason"] = (
                    "Required permission denied."
                )

        else:

            result["allowed"] = (
                self.default_allow
            )

            result["reason"] = (
                "Default access policy."
            )

        self._record_access(
            result
        )

        if result["allowed"]:

            logger.debug(
                "Access allowed: %s -> %s",
                user_id,
                permission,
            )

        else:

            logger.warning(
                "Access denied: %s -> %s",
                user_id,
                permission,
            )

        return result

    # ============================================================
    # SIMPLE ACCESS CHECK
    # ============================================================

    def can_access(
        self,
        user_id: str,
        resource: str,
        action: Optional[
            str
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> bool:
        """
        Return only True or False.
        """

        result = self.check_access(
            user_id=user_id,
            resource=resource,
            action=action,
            context=context,
        )

        return bool(
            result["allowed"]
        )

    # ============================================================
    # REQUIRE ACCESS
    # ============================================================

    def require_access(
        self,
        user_id: str,
        resource: str,
        action: Optional[
            str
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        """
        Raise PermissionError if access
        is not allowed.
        """

        result = self.check_access(
            user_id=user_id,
            resource=resource,
            action=action,
            context=context,
        )

        if not result["allowed"]:

            raise PermissionError(
                result["reason"]
            )

    # ============================================================
    # ACCESS HISTORY
    # ============================================================

    def _record_access(
        self,
        access_result: Dict[str, Any],
    ) -> None:
        """
        Record an access decision.
        """

        self.access_history.append(
            access_result.copy()
        )

        if (
            len(self.access_history)
            > self.max_history
        ):

            overflow = (
                len(self.access_history)
                - self.max_history
            )

            self.access_history = (
                self.access_history[
                    overflow:
                ]
            )

    def get_access_history(
        self,
        user_id: Optional[
            str
        ] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return access history.

        Optionally filter by user.
        """

        history = self.access_history

        if user_id:

            history = [
                item
                for item in history
                if item.get(
                    "user_id"
                ) == user_id
            ]

        return history[
            -max(1, limit):
        ]

    def clear_access_history(
        self,
    ) -> None:
        """
        Clear stored access history.
        """

        self.access_history.clear()

    # ============================================================
    # POLICY INFORMATION
    # ============================================================

    def get_policy(
        self,
        resource: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Get a resource policy.
        """

        policy = self.policies.get(
            resource
        )

        if policy is None:
            return None

        return policy.copy()

    def get_policies(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return all registered policies.
        """

        return {
            resource: policy.copy()
            for (
                resource,
                policy,
            )
            in self.policies.items()
        }

    # ============================================================
    # RULE INFORMATION
    # ============================================================

    def get_rules(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Return access rules.
        """

        return [
            rule.copy()
            for rule in self.rules
        ]

    # ============================================================
    # DEFAULT POLICIES
    # ============================================================

    def register_default_policies(
        self,
    ) -> None:
        """
        Register common RENIX security policies.
        """

        default_permissions = [
            "computer.control",
            "files.read",
            "files.write",
            "files.delete",
            "browser.search",
            "browser.download",
            "coding.execute",
            "terminal.execute",
            "automation.execute",
            "system.settings",
            "system.shutdown",
            "security.manage",
            "devices.control",
        ]

        for permission in (
            default_permissions
        ):

            self.register_policy(
                resource=permission,
                required_permission=permission,
            )

        logger.info(
            "Default access policies registered."
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return access control status.
        """

        return {
            "enabled": self.enabled,
            "default_allow": (
                self.default_allow
            ),
            "policies": len(
                self.policies
            ),
            "rules": len(
                self.rules
            ),
            "access_history": len(
                self.access_history
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shut down access control.
        """

        logger.info(
            "Shutting down AccessControl..."
        )

        self.clear_access_history()

        logger.info(
            "AccessControl shutdown complete."
        )


AccessController = AccessControl
