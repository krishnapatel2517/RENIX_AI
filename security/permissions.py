"""
RENIX Permissions System
========================

Role-based permission management for RENIX.

Features:

* Role-based access control
* Permission registration
* User permissions
* Role permissions
* Permission inheritance
* Temporary permissions
* Permission checking
* Permission revocation
  """

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("RENIX.Permissions")

class PermissionsManager:
    """
    Role and permission management system for RENIX.

    Permission format examples:

        system.shutdown
        computer.control
        files.delete
        browser.download
        security.manage
        automation.execute
    """

    # ============================================================
    # DEFAULT ROLES
    # ============================================================

    DEFAULT_ROLES = {
        "guest": {
            "description": (
                "Basic limited access."
            ),
            "permissions": {
                "system.status",
                "system.info",
                "conversation.basic",
            },
        },

        "user": {
            "description": (
                "Standard RENIX user."
            ),
            "permissions": {
                "system.status",
                "system.info",
                "conversation.basic",
                "computer.control",
                "files.read",
                "files.write",
                "browser.search",
                "browser.read",
                "media.control",
                "personal.read",
                "personal.write",
            },
        },

        "developer": {
            "description": (
                "Development and advanced control."
            ),
            "permissions": {
                "system.status",
                "system.info",
                "system.monitor",
                "system.settings",
                "computer.control",
                "files.read",
                "files.write",
                "files.delete",
                "browser.search",
                "browser.read",
                "browser.download",
                "coding.read",
                "coding.write",
                "coding.execute",
                "terminal.execute",
                "automation.create",
                "automation.execute",
                "plugins.manage",
            },
        },

        "admin": {
            "description": (
                "Full RENIX administrative access."
            ),
            "permissions": {
                "*",
            },
        },
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize permission manager.
        """

        self.config = config or {}

        self.enabled = self.config.get(
            "enabled",
            True,
        )

        self.roles: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.user_roles: Dict[
            str,
            Set[str]
        ] = {}

        self.user_permissions: Dict[
            str,
            Set[str]
        ] = {}

        self.denied_permissions: Dict[
            str,
            Set[str]
        ] = {}

        self.temporary_permissions: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

        self._initialize_default_roles()

        logger.info(
            "PermissionsManager initialized."
        )

    # ============================================================
    # INITIALIZE DEFAULT ROLES
    # ============================================================

    def _initialize_default_roles(
        self,
    ) -> None:
        """
        Load RENIX default roles.
        """

        for (
            role_name,
            role_data,
        ) in self.DEFAULT_ROLES.items():

            self.roles[role_name] = {
                "description": role_data[
                    "description"
                ],
                "permissions": set(
                    role_data[
                        "permissions"
                    ]
                ),
                "created_at": (
                    datetime.now().isoformat()
                ),
            }

    # ============================================================
    # CREATE ROLE
    # ============================================================

    def create_role(
        self,
        role_name: str,
        permissions: Optional[
            List[str]
        ] = None,
        description: str = "",
    ) -> bool:
        """
        Create a custom role.
        """

        if not role_name:
            return False

        if role_name in self.roles:

            logger.warning(
                "Role already exists: %s",
                role_name,
            )

            return False

        self.roles[role_name] = {
            "description": description,
            "permissions": set(
                permissions or []
            ),
            "created_at": (
                datetime.now().isoformat()
            ),
        }

        logger.info(
            "Role created: %s",
            role_name,
        )

        return True

    # ============================================================
    # DELETE ROLE
    # ============================================================

    def delete_role(
        self,
        role_name: str,
    ) -> bool:
        """
        Delete a role.
        """

        if role_name not in self.roles:
            return False

        # Protect default roles.
        if role_name in self.DEFAULT_ROLES:

            logger.warning(
                "Cannot delete default role: %s",
                role_name,
            )

            return False

        del self.roles[role_name]

        # Remove role from users.
        for roles in self.user_roles.values():

            roles.discard(
                role_name
            )

        logger.info(
            "Role deleted: %s",
            role_name,
        )

        return True

    # ============================================================
    # ADD ROLE PERMISSION
    # ============================================================

    def add_role_permission(
        self,
        role_name: str,
        permission: str,
    ) -> bool:
        """
        Add permission to a role.
        """

        role = self.roles.get(
            role_name
        )

        if role is None:
            return False

        role[
            "permissions"
        ].add(
            permission
        )

        return True

    # ============================================================
    # REMOVE ROLE PERMISSION
    # ============================================================

    def remove_role_permission(
        self,
        role_name: str,
        permission: str,
    ) -> bool:
        """
        Remove permission from a role.
        """

        role = self.roles.get(
            role_name
        )

        if role is None:
            return False

        role[
            "permissions"
        ].discard(
            permission
        )

        return True

    # ============================================================
    # ASSIGN ROLE
    # ============================================================

    def assign_role(
        self,
        user_id: str,
        role_name: str,
    ) -> bool:
        """
        Assign a role to a user.
        """

        if role_name not in self.roles:

            logger.warning(
                "Unknown role: %s",
                role_name,
            )

            return False

        if user_id not in self.user_roles:

            self.user_roles[
                user_id
            ] = set()

        self.user_roles[
            user_id
        ].add(
            role_name
        )

        logger.info(
            "Role '%s' assigned to user '%s'.",
            role_name,
            user_id,
        )

        return True

    # ============================================================
    # REMOVE ROLE
    # ============================================================

    def remove_role(
        self,
        user_id: str,
        role_name: str,
    ) -> bool:
        """
        Remove a role from a user.
        """

        roles = self.user_roles.get(
            user_id
        )

        if roles is None:
            return False

        if role_name not in roles:
            return False

        roles.remove(
            role_name
        )

        logger.info(
            "Role '%s' removed from user '%s'.",
            role_name,
            user_id,
        )

        return True

    # ============================================================
    # GRANT USER PERMISSION
    # ============================================================

    def grant_permission(
        self,
        user_id: str,
        permission: str,
    ) -> bool:
        """
        Grant a direct permission to a user.
        """

        if user_id not in self.user_permissions:

            self.user_permissions[
                user_id
            ] = set()

        self.user_permissions[
            user_id
        ].add(
            permission
        )

        logger.info(
            "Permission '%s' granted to '%s'.",
            permission,
            user_id,
        )

        return True

    # ============================================================
    # REVOKE USER PERMISSION
    # ============================================================

    def revoke_permission(
        self,
        user_id: str,
        permission: str,
    ) -> bool:
        """
        Revoke a directly granted permission.
        """

        permissions = (
            self.user_permissions.get(
                user_id
            )
        )

        if permissions is None:
            return False

        if permission not in permissions:
            return False

        permissions.remove(
            permission
        )

        logger.info(
            "Permission '%s' revoked from '%s'.",
            permission,
            user_id,
        )

        return True

    # ============================================================
    # DENY PERMISSION
    # ============================================================

    def deny_permission(
        self,
        user_id: str,
        permission: str,
    ) -> bool:
        """
        Explicitly deny a permission.

        Denied permissions override role permissions.
        """

        if user_id not in self.denied_permissions:

            self.denied_permissions[
                user_id
            ] = set()

        self.denied_permissions[
            user_id
        ].add(
            permission
        )

        logger.info(
            "Permission '%s' denied for '%s'.",
            permission,
            user_id,
        )

        return True

    # ============================================================
    # REMOVE DENIAL
    # ============================================================

    def remove_denial(
        self,
        user_id: str,
        permission: str,
    ) -> bool:
        """
        Remove an explicit permission denial.
        """

        denied = self.denied_permissions.get(
            user_id
        )

        if denied is None:
            return False

        if permission not in denied:
            return False

        denied.remove(
            permission
        )

        return True

    # ============================================================
    # TEMPORARY PERMISSION
    # ============================================================

    def grant_temporary_permission(
        self,
        user_id: str,
        permission: str,
        duration_seconds: int,
    ) -> bool:
        """
        Grant permission temporarily.
        """

        if duration_seconds <= 0:
            return False

        expires_at = (
            datetime.now()
            + timedelta(
                seconds=duration_seconds
            )
        )

        if user_id not in (
            self.temporary_permissions
        ):

            self.temporary_permissions[
                user_id
            ] = []

        self.temporary_permissions[
            user_id
        ].append(
            {
                "permission": permission,
                "expires_at": expires_at,
            }
        )

        logger.info(
            "Temporary permission '%s' granted "
            "to '%s'.",
            permission,
            user_id,
        )

        return True

    # ============================================================
    # CLEAN TEMPORARY PERMISSIONS
    # ============================================================

    def cleanup_temporary_permissions(
        self,
    ) -> int:
        """
        Remove expired temporary permissions.
        """

        removed = 0
        now = datetime.now()

        for (
            user_id,
            permissions,
        ) in list(
            self.temporary_permissions.items()
        ):

            valid_permissions = []

            for permission_data in permissions:

                expires_at = permission_data.get(
                    "expires_at"
                )

                if (
                    expires_at
                    and expires_at > now
                ):

                    valid_permissions.append(
                        permission_data
                    )

                else:

                    removed += 1

            if valid_permissions:

                self.temporary_permissions[
                    user_id
                ] = valid_permissions

            else:

                del self.temporary_permissions[
                    user_id
                ]

        return removed

    # ============================================================
    # GET ROLE PERMISSIONS
    # ============================================================

    def get_role_permissions(
        self,
        role_name: str,
    ) -> Set[str]:
        """
        Get permissions belonging to a role.
        """

        role = self.roles.get(
            role_name
        )

        if role is None:
            return set()

        return set(
            role["permissions"]
        )

    # ============================================================
    # GET USER PERMISSIONS
    # ============================================================

    def get_user_permissions(
        self,
        user_id: str,
    ) -> Set[str]:
        """
        Get all effective permissions
        for a user.
        """

        permissions: Set[str] = set()

        # Role permissions.
        for role_name in (
            self.user_roles.get(
                user_id,
                set(),
            )
        ):

            role_permissions = (
                self.get_role_permissions(
                    role_name
                )
            )

            permissions.update(
                role_permissions
            )

        # Direct permissions.
        permissions.update(
            self.user_permissions.get(
                user_id,
                set(),
            )
        )

        # Temporary permissions.
        now = datetime.now()

        for permission_data in (
            self.temporary_permissions.get(
                user_id,
                [],
            )
        ):

            expires_at = permission_data.get(
                "expires_at"
            )

            if (
                expires_at is None
                or expires_at > now
            ):

                permissions.add(
                    permission_data[
                        "permission"
                    ]
                )

        return permissions

    # ============================================================
    # PERMISSION MATCHING
    # ============================================================

    @staticmethod
    def _matches_permission(
        granted: str,
        required: str,
    ) -> bool:
        """
        Determine whether a granted permission
        satisfies a required permission.

        Examples:

            "*" matches everything
            "files.*" matches "files.delete"
            "files.read" matches only "files.read"
        """

        if granted == "*":
            return True

        if granted == required:
            return True

        if granted.endswith(".*"):

            prefix = granted[:-1]

            return required.startswith(
                prefix
            )

        return False

    # ============================================================
    # CHECK PERMISSION
    # ============================================================

    def has_permission(
        self,
        user_id: str,
        permission: str,
    ) -> bool:
        """
        Check whether a user has permission.
        """

        if not self.enabled:
            return True

        # Explicit denial always wins.
        denied_permissions = (
            self.denied_permissions.get(
                user_id,
                set(),
            )
        )

        for denied in denied_permissions:

            if self._matches_permission(
                denied,
                permission,
            ):

                return False

        # Check granted permissions.
        permissions = (
            self.get_user_permissions(
                user_id
            )
        )

        for granted in permissions:

            if self._matches_permission(
                granted,
                permission,
            ):

                return True

        return False

    # ============================================================
    # REQUIRE PERMISSION
    # ============================================================

    def require_permission(
        self,
        user_id: str,
        permission: str,
    ) -> None:
        """
        Raise PermissionError if permission
        is not granted.
        """

        if not self.has_permission(
            user_id,
            permission,
        ):

            raise PermissionError(
                f"User '{user_id}' does not have "
                f"permission '{permission}'."
            )

    # ============================================================
    # CHECK MULTIPLE PERMISSIONS
    # ============================================================

    def has_all_permissions(
        self,
        user_id: str,
        permissions: List[str],
    ) -> bool:
        """
        Check whether user has all permissions.
        """

        return all(
            self.has_permission(
                user_id,
                permission,
            )
            for permission in permissions
        )

    def has_any_permission(
        self,
        user_id: str,
        permissions: List[str],
    ) -> bool:
        """
        Check whether user has at least
        one permission.
        """

        return any(
            self.has_permission(
                user_id,
                permission,
            )
            for permission in permissions
        )

    # ============================================================
    # USER ROLES
    # ============================================================

    def get_user_roles(
        self,
        user_id: str,
    ) -> List[str]:
        """
        Return roles assigned to a user.
        """

        return list(
            self.user_roles.get(
                user_id,
                set(),
            )
        )

    # ============================================================
    # REMOVE USER
    # ============================================================

    def remove_user(
        self,
        user_id: str,
    ) -> bool:
        """
        Remove all permission information
        associated with a user.
        """

        removed = False

        if user_id in self.user_roles:

            del self.user_roles[
                user_id
            ]

            removed = True

        if user_id in self.user_permissions:

            del self.user_permissions[
                user_id
            ]

            removed = True

        if user_id in self.denied_permissions:

            del self.denied_permissions[
                user_id
            ]

            removed = True

        if user_id in (
            self.temporary_permissions
        ):

            del self.temporary_permissions[
                user_id
            ]

            removed = True

        return removed

    # ============================================================
    # LIST ROLES
    # ============================================================

    def get_roles(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return available roles.
        """

        result = {}

        for (
            role_name,
            role_data,
        ) in self.roles.items():

            result[role_name] = {
                "description": role_data[
                    "description"
                ],
                "permissions": list(
                    role_data[
                        "permissions"
                    ]
                ),
            }

        return result

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return permissions system status.
        """

        return {
            "enabled": self.enabled,
            "roles": len(
                self.roles
            ),
            "users_with_roles": len(
                self.user_roles
            ),
            "users_with_permissions": len(
                self.user_permissions
            ),
            "temporary_permission_users": len(
                self.temporary_permissions
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Safely shut down permissions manager.
        """

        logger.info(
            "Shutting down PermissionsManager..."
        )

        self.temporary_permissions.clear()

        logger.info(
            "PermissionsManager shutdown complete."
        )


PermissionManager = PermissionsManager
