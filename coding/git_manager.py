"""
RENIX Git Manager
=================

Provides Git repository management for RENIX's coding system.

Responsibilities:
    - Detect Git repositories
    - Read repository status
    - Initialize repositories
    - Create branches
    - Switch branches
    - Stage files
    - Commit changes
    - Push and pull changes
    - Read commit history
    - Inspect diffs
    - Manage remotes

The manager uses subprocess without shell=True and treats Git
arguments as individual arguments rather than shell commands.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class GitResult:
    """Structured result of a Git operation."""

    success: bool
    operation: str
    command: list[str]
    return_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    repository: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "repository": self.repository,
        }


@dataclass
class GitStatus:
    """Repository working-tree status."""

    branch: str | None
    ahead: int
    behind: int
    staged: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    renamed: list[str] = field(default_factory=list)
    conflicted: list[str] = field(default_factory=list)
    clean: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "ahead": self.ahead,
            "behind": self.behind,
            "staged": self.staged,
            "modified": self.modified,
            "deleted": self.deleted,
            "untracked": self.untracked,
            "renamed": self.renamed,
            "conflicted": self.conflicted,
            "clean": self.clean,
        }


@dataclass
class GitCommit:
    """Git commit information."""

    hash: str
    short_hash: str
    author: str
    email: str
    date: str
    subject: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "short_hash": self.short_hash,
            "author": self.author,
            "email": self.email,
            "date": self.date,
            "subject": self.subject,
        }


class GitManager:
    """Git repository management engine for RENIX."""

    DEFAULT_TIMEOUT = 120.0
    MAX_OUTPUT_SIZE = 4 * 1024 * 1024

    def __init__(
        self,
        repository: str | os.PathLike[str] | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:

        self.repository = (
            Path(repository)
            .expanduser()
            .resolve()
            if repository is not None
            else Path.cwd().resolve()
        )

        self.timeout = timeout

    # =========================================================
    # REPOSITORY DISCOVERY
    # =========================================================

    def is_git_repository(
        self,
        path: str | os.PathLike[str] | None = None,
    ) -> bool:

        target = (
            self.repository
            if path is None
            else Path(path)
            .expanduser()
            .resolve()
        )

        git_path = target / ".git"

        return git_path.exists()

    def find_repository_root(
        self,
        path: str | os.PathLike[str] | None = None,
    ) -> Path | None:

        target = (
            self.repository
            if path is None
            else Path(path)
            .expanduser()
            .resolve()
        )

        if target.is_file():
            target = target.parent

        for current in (
            target,
            *target.parents,
        ):

            if (
                current / ".git"
            ).exists():

                return current

        return None

    def require_repository(self) -> Path:
        """Return repository root or raise an error."""

        root = self.find_repository_root()

        if root is None:
            raise RuntimeError(
                f"Not a Git repository: "
                f"{self.repository}"
            )

        return root

    # =========================================================
    # BASIC GIT EXECUTION
    # =========================================================

    def run(
        self,
        arguments: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        timeout: float | None = None,
    ) -> GitResult:

        root = (
            Path(cwd)
            .expanduser()
            .resolve()
            if cwd is not None
            else self.repository
        )

        command = [
            "git",
            *[
                str(argument)
                for argument in arguments
            ],
        ]

        start = time.perf_counter()

        try:

            process = subprocess.run(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=(
                    self.timeout
                    if timeout is None
                    else timeout
                ),
                shell=False,
                check=False,
            )

            duration = (
                time.perf_counter()
                - start
            )

            return GitResult(
                success=(
                    process.returncode == 0
                ),
                operation=(
                    arguments[0]
                    if arguments
                    else "git"
                ),
                command=command,
                return_code=process.returncode,
                stdout=self._limit_output(
                    process.stdout
                ),
                stderr=self._limit_output(
                    process.stderr
                ),
                duration_seconds=duration,
                repository=str(root),
            )

        except subprocess.TimeoutExpired as exc:

            return GitResult(
                success=False,
                operation=(
                    arguments[0]
                    if arguments
                    else "git"
                ),
                command=command,
                return_code=None,
                stdout=self._convert_output(
                    exc.stdout
                ),
                stderr=(
                    self._convert_output(
                        exc.stderr
                    )
                    or "Git operation timed out."
                ),
                duration_seconds=(
                    time.perf_counter()
                    - start
                ),
                repository=str(root),
            )

        except FileNotFoundError as exc:

            return GitResult(
                success=False,
                operation="git",
                command=command,
                return_code=None,
                stdout="",
                stderr=(
                    "Git executable was not found: "
                    f"{exc}"
                ),
                duration_seconds=(
                    time.perf_counter()
                    - start
                ),
                repository=str(root),
            )

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def init(
        self,
        *,
        path: str | os.PathLike[str] | None = None,
        bare: bool = False,
    ) -> GitResult:

        target = (
            self.repository
            if path is None
            else Path(path)
            .expanduser()
            .resolve()
        )

        target.mkdir(
            parents=True,
            exist_ok=True,
        )

        arguments = ["init"]

        if bare:
            arguments.append("--bare")

        return self.run(
            arguments,
            cwd=target,
        )

    # =========================================================
    # STATUS
    # =========================================================

    def status(
        self,
    ) -> GitStatus:

        root = self.require_repository()

        result = self.run(
            [
                "status",
                "--porcelain=v1",
                "--branch",
            ],
            cwd=root,
        )

        if not result.success:
            raise RuntimeError(
                result.stderr
                or "Unable to read Git status."
            )

        return self._parse_status(
            result.stdout
        )

    def _parse_status(
        self,
        output: str,
    ) -> GitStatus:

        branch = None
        ahead = 0
        behind = 0

        staged: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []
        untracked: list[str] = []
        renamed: list[str] = []
        conflicted: list[str] = []

        lines = output.splitlines()

        for line in lines:

            if line.startswith(
                "## "
            ):

                branch_info = line[3:]

                branch = (
                    branch_info
                    .split("...", 1)[0]
                    .strip()
                )

                if (
                    "[ahead "
                    in branch_info
                ):

                    try:
                        ahead = int(
                            branch_info
                            .split(
                                "[ahead ",
                                1,
                            )[1]
                            .split("]", 1)[0]
                        )
                    except (
                        ValueError,
                        IndexError,
                    ):
                        ahead = 0

                if (
                    "[behind "
                    in branch_info
                ):

                    try:
                        behind = int(
                            branch_info
                            .split(
                                "[behind ",
                                1,
                            )[1]
                            .split("]", 1)[0]
                        )
                    except (
                        ValueError,
                        IndexError,
                    ):
                        behind = 0

                continue

            if not line:
                continue

            if line.startswith(
                "?? "
            ):

                untracked.append(
                    line[3:]
                )

                continue

            if len(line) < 3:
                continue

            index_status = line[0]
            worktree_status = line[1]
            filename = line[3:]

            if (
                index_status == "U"
                or worktree_status == "U"
                or (
                    index_status == "A"
                    and worktree_status == "A"
                )
                or (
                    index_status == "D"
                    and worktree_status == "D"
                )
            ):

                conflicted.append(
                    filename
                )

            if index_status in {
                "A",
                "M",
                "D",
                "R",
                "C",
            }:

                if index_status == "A":
                    staged.append(filename)

                elif index_status == "M":
                    staged.append(filename)

                elif index_status == "D":
                    staged.append(filename)

                elif index_status == "R":
                    renamed.append(filename)

                else:
                    staged.append(filename)

            if worktree_status == "M":
                modified.append(filename)

            elif worktree_status == "D":
                deleted.append(filename)

            elif worktree_status == "R":
                renamed.append(filename)

        return GitStatus(
            branch=branch,
            ahead=ahead,
            behind=behind,
            staged=staged,
            modified=modified,
            deleted=deleted,
            untracked=untracked,
            renamed=renamed,
            conflicted=conflicted,
            clean=(
                not staged
                and not modified
                and not deleted
                and not untracked
                and not renamed
                and not conflicted
            ),
        )

    # =========================================================
    # BRANCHES
    # =========================================================

    def current_branch(self) -> str | None:

        result = self.run(
            [
                "branch",
                "--show-current",
            ]
        )

        if not result.success:
            return None

        branch = result.stdout.strip()

        return branch or None

    def list_branches(
        self,
        *,
        remote: bool = False,
    ) -> list[str]:

        arguments = [
            "branch",
        ]

        if remote:
            arguments.append("-r")

        result = self.run(
            arguments
        )

        if not result.success:
            raise RuntimeError(
                result.stderr
                or "Unable to list branches."
            )

        branches: list[str] = []

        for line in result.stdout.splitlines():

            cleaned = line.strip()

            if cleaned.startswith("* "):
                cleaned = cleaned[2:].strip()

            if cleaned:
                branches.append(
                    cleaned
                )

        return branches

    def create_branch(
        self,
        branch_name: str,
        *,
        checkout: bool = False,
    ) -> GitResult:

        self._validate_branch_name(
            branch_name
        )

        arguments = [
            "checkout",
            "-b",
            branch_name,
        ]

        if not checkout:
            arguments = [
                "branch",
                branch_name,
            ]

        return self.run(
            arguments
        )

    def switch_branch(
        self,
        branch_name: str,
        *,
        create: bool = False,
    ) -> GitResult:

        self._validate_branch_name(
            branch_name
        )

        if create:

            arguments = [
                "switch",
                "-c",
                branch_name,
            ]

        else:

            arguments = [
                "switch",
                branch_name,
            ]

        return self.run(
            arguments
        )

    def delete_branch(
        self,
        branch_name: str,
        *,
        force: bool = False,
    ) -> GitResult:

        self._validate_branch_name(
            branch_name
        )

        flag = "-D" if force else "-d"

        return self.run(
            [
                "branch",
                flag,
                branch_name,
            ]
        )

    # =========================================================
    # STAGING
    # =========================================================

    def stage(
        self,
        paths: Sequence[str] | None = None,
    ) -> GitResult:

        if not paths:
            paths = ["."]

        safe_paths = [
            self._validate_path_argument(
                path
            )
            for path in paths
        ]

        return self.run(
            [
                "add",
                "--",
                *safe_paths,
            ]
        )

    def unstage(
        self,
        paths: Sequence[str] | None = None,
    ) -> GitResult:

        if not paths:
            paths = ["."]

        safe_paths = [
            self._validate_path_argument(
                path
            )
            for path in paths
        ]

        return self.run(
            [
                "restore",
                "--staged",
                "--",
                *safe_paths,
            ]
        )

    # =========================================================
    # COMMIT
    # =========================================================

    def commit(
        self,
        message: str,
    ) -> GitResult:

        message = message.strip()

        if not message:
            raise ValueError(
                "Commit message cannot be empty."
            )

        if len(message) > 1000:
            raise ValueError(
                "Commit message is too long."
            )

        return self.run(
            [
                "commit",
                "-m",
                message,
            ]
        )

    # =========================================================
    # DIFF
    # =========================================================

    def diff(
        self,
        *,
        staged: bool = False,
        paths: Sequence[str] | None = None,
    ) -> str:

        arguments = [
            "diff",
        ]

        if staged:
            arguments.append(
                "--cached"
            )

        if paths:
            arguments.extend(
                [
                    "--",
                    *[
                        self._validate_path_argument(
                            path
                        )
                        for path in paths
                    ],
                ]
            )

        result = self.run(
            arguments
        )

        if not result.success:
            raise RuntimeError(
                result.stderr
                or "Unable to read Git diff."
            )

        return result.stdout

    # =========================================================
    # LOG
    # =========================================================

    def log(
        self,
        *,
        limit: int = 20,
    ) -> list[GitCommit]:

        if limit <= 0:
            return []

        limit = min(
            limit,
            500,
        )

        separator = (
            "%H%x1f"
            "%h%x1f"
            "%an%x1f"
            "%ae%x1f"
            "%ad%x1f"
            "%s%x1e"
        )

        result = self.run(
            [
                "log",
                f"-n{limit}",
                f"--format={separator}",
                "--date=iso-strict",
            ]
        )

        if not result.success:
            raise RuntimeError(
                result.stderr
                or "Unable to read Git history."
            )

        commits: list[GitCommit] = []

        for record in result.stdout.split(
            "\x1e"
        ):

            record = record.strip(
                "\n"
            )

            if not record:
                continue

            fields = record.split(
                "\x1f"
            )

            if len(fields) != 6:
                continue

            commits.append(
                GitCommit(
                    hash=fields[0],
                    short_hash=fields[1],
                    author=fields[2],
                    email=fields[3],
                    date=fields[4],
                    subject=fields[5],
                )
            )

        return commits

    # =========================================================
    # REMOTES
    # =========================================================

    def list_remotes(
        self,
    ) -> dict[str, str]:

        result = self.run(
            [
                "remote",
                "-v",
            ]
        )

        if not result.success:
            raise RuntimeError(
                result.stderr
                or "Unable to read Git remotes."
            )

        remotes: dict[str, str] = {}

        for line in result.stdout.splitlines():

            parts = line.split()

            if len(parts) < 2:
                continue

            name = parts[0]
            url = parts[1]

            if name not in remotes:
                remotes[name] = url

        return remotes

    def add_remote(
        self,
        name: str,
        url: str,
    ) -> GitResult:

        self._validate_remote_name(
            name
        )

        self._validate_remote_url(
            url
        )

        return self.run(
            [
                "remote",
                "add",
                name,
                url,
            ]
        )

    def remove_remote(
        self,
        name: str,
    ) -> GitResult:

        self._validate_remote_name(
            name
        )

        return self.run(
            [
                "remote",
                "remove",
                name,
            ]
        )

    # =========================================================
    # SYNC
    # =========================================================

    def fetch(
        self,
        remote: str | None = None,
    ) -> GitResult:

        arguments = [
            "fetch",
        ]

        if remote:
            self._validate_remote_name(
                remote
            )
            arguments.append(
                remote
            )

        return self.run(
            arguments
        )

    def pull(
        self,
        remote: str | None = None,
        branch: str | None = None,
    ) -> GitResult:

        arguments = [
            "pull",
        ]

        if remote:
            self._validate_remote_name(
                remote
            )
            arguments.append(
                remote
            )

        if branch:
            self._validate_branch_name(
                branch
            )
            arguments.append(
                branch
            )

        return self.run(
            arguments
        )

    def push(
        self,
        remote: str | None = None,
        branch: str | None = None,
        *,
        set_upstream: bool = False,
    ) -> GitResult:

        arguments = [
            "push",
        ]

        if set_upstream:
            arguments.extend(
                [
                    "-u",
                ]
            )

        if remote:
            self._validate_remote_name(
                remote
            )
            arguments.append(
                remote
            )

        if branch:
            self._validate_branch_name(
                branch
            )
            arguments.append(
                branch
            )

        return self.run(
            arguments
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        *,
        mode: str = "mixed",
        target: str = "HEAD",
    ) -> GitResult:

        allowed_modes = {
            "soft",
            "mixed",
            "hard",
        }

        if mode not in allowed_modes:
            raise ValueError(
                "Invalid reset mode. "
                "Use soft, mixed, or hard."
            )

        if mode == "hard":
            logger.warning(
                "Hard Git reset requested."
            )

        return self.run(
            [
                "reset",
                f"--{mode}",
                target,
            ]
        )

    # =========================================================
    # TAGS
    # =========================================================

    def create_tag(
        self,
        tag_name: str,
        *,
        message: str | None = None,
    ) -> GitResult:

        self._validate_tag_name(
            tag_name
        )

        arguments = [
            "tag",
        ]

        if message:
            arguments.extend(
                [
                    "-a",
                    tag_name,
                    "-m",
                    message,
                ]
            )

        else:
            arguments.append(
                tag_name
            )

        return self.run(
            arguments
        )

    def list_tags(self) -> list[str]:

        result = self.run(
            [
                "tag",
                "--list",
            ]
        )

        if not result.success:
            raise RuntimeError(
                result.stderr
            )

        return [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]

    # =========================================================
    # VALIDATION HELPERS
    # =========================================================

    @staticmethod
    def _validate_branch_name(
        name: str,
    ) -> None:

        if not name.strip():
            raise ValueError(
                "Branch name cannot be empty."
            )

        if (
            name.startswith("-")
            or " " in name
            or ".." in name
            or name.endswith(".")
            or name.startswith(".")
            or "@{" in name
        ):
            raise ValueError(
                f"Invalid Git branch name: {name}"
            )

    @staticmethod
    def _validate_tag_name(
        name: str,
    ) -> None:

        if not name.strip():
            raise ValueError(
                "Tag name cannot be empty."
            )

        if (
            name.startswith("-")
            or " " in name
            or ".." in name
        ):
            raise ValueError(
                f"Invalid Git tag name: {name}"
            )

    @staticmethod
    def _validate_remote_name(
        name: str,
    ) -> None:

        if not name.strip():
            raise ValueError(
                "Remote name cannot be empty."
            )

        if (
            name.startswith("-")
            or any(
                char.isspace()
                for char in name
            )
        ):
            raise ValueError(
                f"Invalid remote name: {name}"
            )

    @staticmethod
    def _validate_remote_url(
        url: str,
    ) -> None:

        if not url.strip():
            raise ValueError(
                "Remote URL cannot be empty."
            )

        if any(
            character in url
            for character in (
                "\n",
                "\r",
            )
        ):
            raise ValueError(
                "Remote URL contains invalid characters."
            )

    @staticmethod
    def _validate_path_argument(
        path: str,
    ) -> str:

        value = str(path)

        if (
            "\x00" in value
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError(
                "Invalid Git path argument."
            )

        return value

    # =========================================================
    # OUTPUT HELPERS
    # =========================================================

    @staticmethod
    def _convert_output(
        output: str | bytes | None,
    ) -> str:

        if output is None:
            return ""

        if isinstance(
            output,
            bytes,
        ):
            return output.decode(
                "utf-8",
                errors="replace",
            )

        return str(output)

    def _limit_output(
        self,
        output: str | bytes | None,
    ) -> str:

        text = self._convert_output(
            output
        )

        if len(text) <= self.MAX_OUTPUT_SIZE:
            return text

        return (
            text[:self.MAX_OUTPUT_SIZE]
            + "\n\n"
            "[RENIX] Git output truncated."
        )


__all__ = [
    "GitManager",
    "GitResult",
    "GitStatus",
    "GitCommit",
]


