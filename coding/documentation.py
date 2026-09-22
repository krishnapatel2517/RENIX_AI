"""
RENIX Documentation Manager
===========================

Generates and manages documentation for RENIX coding projects.

Responsibilities:
    - Extract documentation from Python source files
    - Generate module/class/function documentation
    - Build Markdown documentation
    - Generate project API documentation
    - Read docstrings
    - Create README sections
    - Create documentation indexes
    - Save generated documentation safely

This module focuses on documentation generation and does not
execute arbitrary project code.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ParameterDocumentation:
    """Documentation for a function parameter."""

    name: str
    annotation: str | None = None
    default: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "annotation": self.annotation,
            "default": self.default,
            "description": self.description,
        }


@dataclass
class FunctionDocumentation:
    """Documentation extracted from a function."""

    name: str
    qualified_name: str
    docstring: str
    parameters: list[ParameterDocumentation] = field(
        default_factory=list
    )
    return_annotation: str | None = None
    is_async: bool = False
    is_method: bool = False
    decorators: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "docstring": self.docstring,
            "parameters": [
                parameter.to_dict()
                for parameter in self.parameters
            ],
            "return_annotation": self.return_annotation,
            "is_async": self.is_async,
            "is_method": self.is_method,
            "decorators": self.decorators,
        }


@dataclass
class ClassDocumentation:
    """Documentation extracted from a class."""

    name: str
    qualified_name: str
    docstring: str
    bases: list[str] = field(
        default_factory=list
    )
    methods: list[FunctionDocumentation] = field(
        default_factory=list
    )
    decorators: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "docstring": self.docstring,
            "bases": self.bases,
            "methods": [
                method.to_dict()
                for method in self.methods
            ],
            "decorators": self.decorators,
        }


@dataclass
class ModuleDocumentation:
    """Documentation extracted from a Python module."""

    name: str
    path: str
    docstring: str
    classes: list[ClassDocumentation] = field(
        default_factory=list
    )
    functions: list[FunctionDocumentation] = field(
        default_factory=list
    )
    constants: list[str] = field(
        default_factory=list
    )
    imports: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "docstring": self.docstring,
            "classes": [
                item.to_dict()
                for item in self.classes
            ],
            "functions": [
                item.to_dict()
                for item in self.functions
            ],
            "constants": self.constants,
            "imports": self.imports,
        }


@dataclass
class DocumentationResult:
    """Result of a documentation operation."""

    success: bool
    output_file: str | None
    modules_processed: int
    functions_documented: int
    classes_documented: int
    errors: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output_file": self.output_file,
            "modules_processed": self.modules_processed,
            "functions_documented": self.functions_documented,
            "classes_documented": self.classes_documented,
            "errors": self.errors,
        }


class DocumentationManager:
    """Documentation generation engine for RENIX."""

    IGNORE_DIRECTORIES = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }

    def __init__(
        self,
        project_root: str | Path | None = None,
    ) -> None:

        self.project_root = (
            Path(project_root)
            .expanduser()
            .resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )

    # =========================================================
    # MODULE ANALYSIS
    # =========================================================

    def analyze_file(
        self,
        file_path: str | Path,
    ) -> ModuleDocumentation:

        path = (
            Path(file_path)
            .expanduser()
            .resolve()
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Python file not found: {path}"
            )

        if path.suffix != ".py":
            raise ValueError(
                f"Expected Python file: {path}"
            )

        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        try:

            tree = ast.parse(
                source,
                filename=str(path),
            )

        except SyntaxError as exc:

            raise ValueError(
                f"Unable to parse {path}: {exc}"
            ) from exc

        module_name = self._module_name(
            path
        )

        module_docstring = (
            ast.get_docstring(tree)
            or ""
        )

        classes: list[
            ClassDocumentation
        ] = []

        functions: list[
            FunctionDocumentation
        ] = []

        constants: list[str] = []
        imports: list[str] = []

        for node in tree.body:

            if isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):

                imports.extend(
                    self._extract_imports(
                        node
                    )
                )

            elif isinstance(
                node,
                ast.ClassDef,
            ):

                classes.append(
                    self._analyze_class(
                        node,
                        parent_name=None,
                    )
                )

            elif isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):

                functions.append(
                    self._analyze_function(
                        node,
                        parent_name=None,
                    )
                )

            elif isinstance(
                node,
                ast.Assign,
            ):

                constants.extend(
                    self._extract_constants(
                        node
                    )
                )

            elif isinstance(
                node,
                ast.AnnAssign,
            ):

                constant = (
                    self._extract_ann_constant(
                        node
                    )
                )

                if constant:
                    constants.append(
                        constant
                    )

        return ModuleDocumentation(
            name=module_name,
            path=str(path),
            docstring=module_docstring,
            classes=classes,
            functions=functions,
            constants=constants,
            imports=imports,
        )

    def _analyze_class(
        self,
        node: ast.ClassDef,
        parent_name: str | None,
    ) -> ClassDocumentation:

        qualified_name = (
            f"{parent_name}.{node.name}"
            if parent_name
            else node.name
        )

        methods: list[
            FunctionDocumentation
        ] = []

        nested_classes: list[
            ClassDocumentation
        ] = []

        for child in node.body:

            if isinstance(
                child,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):

                methods.append(
                    self._analyze_function(
                        child,
                        parent_name=qualified_name,
                        is_method=True,
                    )
                )

            elif isinstance(
                child,
                ast.ClassDef,
            ):

                nested_classes.append(
                    self._analyze_class(
                        child,
                        parent_name=qualified_name,
                    )
                )

        # Flatten nested class methods into the main
        # documentation model by keeping the nested class
        # as a documented class in the same class list
        # when rendered.

        documentation = ClassDocumentation(
            name=node.name,
            qualified_name=qualified_name,
            docstring=(
                ast.get_docstring(node)
                or ""
            ),
            bases=[
                self._annotation_to_string(
                    base
                )
                for base in node.bases
            ],
            methods=methods,
            decorators=[
                self._annotation_to_string(
                    decorator
                )
                for decorator in node.decorator_list
            ],
        )

        # Store nested classes dynamically so the renderer
        # can include them without changing the public model.
        if nested_classes:
            setattr(
                documentation,
                "nested_classes",
                nested_classes,
            )

        return documentation

    def _analyze_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_name: str | None,
        is_method: bool = False,
    ) -> FunctionDocumentation:

        qualified_name = (
            f"{parent_name}.{node.name}"
            if parent_name
            else node.name
        )

        parameters: list[
            ParameterDocumentation
        ] = []

        all_args = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]

        defaults = (
            [None] * (
                len(node.args.posonlyargs)
                + len(node.args.args)
                - len(node.args.defaults)
            )
            + list(node.args.defaults)
        )

        default_map: dict[str, str] = {}

        positional_args = [
            *node.args.posonlyargs,
            *node.args.args,
        ]

        for argument, default in zip(
            positional_args,
            defaults,
        ):

            if default is not None:
                default_map[
                    argument.arg
                ] = self._annotation_to_string(
                    default
                )

        for argument in all_args:

            annotation = None

            if argument.annotation is not None:
                annotation = (
                    self._annotation_to_string(
                        argument.annotation
                    )
                )

            parameters.append(
                ParameterDocumentation(
                    name=argument.arg,
                    annotation=annotation,
                    default=default_map.get(
                        argument.arg
                    ),
                )
            )

        if node.args.vararg:

            parameters.append(
                ParameterDocumentation(
                    name="*"
                    + node.args.vararg.arg,
                    annotation=(
                        self._annotation_to_string(
                            node.args.vararg.annotation
                        )
                        if node.args.vararg.annotation
                        else None
                    ),
                )
            )

        if node.args.kwarg:

            parameters.append(
                ParameterDocumentation(
                    name="**"
                    + node.args.kwarg.arg,
                    annotation=(
                        self._annotation_to_string(
                            node.args.kwarg.annotation
                        )
                        if node.args.kwarg.annotation
                        else None
                    ),
                )
            )

        return_annotation = None

        if node.returns is not None:
            return_annotation = (
                self._annotation_to_string(
                    node.returns
                )
            )

        return FunctionDocumentation(
            name=node.name,
            qualified_name=qualified_name,
            docstring=(
                ast.get_docstring(node)
                or ""
            ),
            parameters=parameters,
            return_annotation=return_annotation,
            is_async=isinstance(
                node,
                ast.AsyncFunctionDef,
            ),
            is_method=is_method,
            decorators=[
                self._annotation_to_string(
                    decorator
                )
                for decorator in node.decorator_list
            ],
        )

    # =========================================================
    # PROJECT ANALYSIS
    # =========================================================

    def analyze_project(
        self,
        root: str | Path | None = None,
    ) -> list[ModuleDocumentation]:

        project_root = (
            self.project_root
            if root is None
            else Path(root)
            .expanduser()
            .resolve()
        )

        if not project_root.is_dir():
            raise NotADirectoryError(
                f"Project directory not found: "
                f"{project_root}"
            )

        modules: list[
            ModuleDocumentation
        ] = []

        for path in sorted(
            project_root.rglob("*.py")
        ):

            if self._should_ignore(
                path,
                project_root,
            ):
                continue

            try:

                modules.append(
                    self.analyze_file(
                        path
                    )
                )

            except Exception as exc:

                logger.warning(
                    "Unable to document %s: %s",
                    path,
                    exc,
                )

        return modules

    # =========================================================
    # MARKDOWN GENERATION
    # =========================================================

    def generate_module_markdown(
        self,
        module: ModuleDocumentation,
    ) -> str:

        lines: list[str] = []

        lines.append(
            f"# `{module.name}`"
        )

        lines.append("")

        lines.append(
            f"**Source:** `{module.path}`"
        )

        lines.append("")

        if module.docstring:

            lines.append(
                self._clean_docstring(
                    module.docstring
                )
            )

            lines.append("")

        if module.imports:

            lines.append(
                "## Imports"
            )

            lines.append("")

            for item in module.imports:
                lines.append(
                    f"- `{item}`"
                )

            lines.append("")

        if module.constants:

            lines.append(
                "## Constants"
            )

            lines.append("")

            for constant in module.constants:
                lines.append(
                    f"- `{constant}`"
                )

            lines.append("")

        if module.classes:

            lines.append(
                "## Classes"
            )

            lines.append("")

            for class_doc in module.classes:

                lines.extend(
                    self._render_class(
                        class_doc
                    )
                )

        if module.functions:

            lines.append(
                "## Functions"
            )

            lines.append("")

            for function in module.functions:

                lines.extend(
                    self._render_function(
                        function,
                        heading_level=3,
                    )
                )

        return "\n".join(
            lines
        ).rstrip() + "\n"

    def _render_class(
        self,
        class_doc: ClassDocumentation,
    ) -> list[str]:

        lines: list[str] = []

        lines.append(
            f"### `{class_doc.qualified_name}`"
        )

        lines.append("")

        if class_doc.bases:

            lines.append(
                "**Bases:** "
                + ", ".join(
                    f"`{base}`"
                    for base in class_doc.bases
                )
            )

            lines.append("")

        if class_doc.decorators:

            lines.append(
                "**Decorators:** "
                + ", ".join(
                    f"`{item}`"
                    for item in class_doc.decorators
                )
            )

            lines.append("")

        if class_doc.docstring:

            lines.append(
                self._clean_docstring(
                    class_doc.docstring
                )
            )

            lines.append("")

        if class_doc.methods:

            lines.append(
                "#### Methods"
            )

            lines.append("")

            for method in class_doc.methods:

                lines.extend(
                    self._render_function(
                        method,
                        heading_level=5,
                    )
                )

        nested = getattr(
            class_doc,
            "nested_classes",
            [],
        )

        for nested_class in nested:

            lines.extend(
                self._render_class(
                    nested_class
                )
            )

        return lines

    def _render_function(
        self,
        function: FunctionDocumentation,
        *,
        heading_level: int = 3,
    ) -> list[str]:

        lines: list[str] = []

        heading = "#" * heading_level

        prefix = (
            "async "
            if function.is_async
            else ""
        )

        signature_parts: list[str] = []

        for parameter in function.parameters:

            value = parameter.name

            if parameter.annotation:
                value += (
                    ": "
                    + parameter.annotation
                )

            if parameter.default:
                value += (
                    " = "
                    + parameter.default
                )

            signature_parts.append(
                value
            )

        signature = (
            f"{prefix}{function.name}"
            f"({', '.join(signature_parts)})"
        )

        if function.return_annotation:
            signature += (
                " -> "
                + function.return_annotation
            )

        lines.append(
            f"{heading} `{signature}`"
        )

        lines.append("")

        if function.decorators:

            lines.append(
                "**Decorators:** "
                + ", ".join(
                    f"`{item}`"
                    for item in function.decorators
                )
            )

            lines.append("")

        if function.docstring:

            lines.append(
                self._clean_docstring(
                    function.docstring
                )
            )

        else:

            lines.append(
                "_No documentation provided._"
            )

        lines.append("")

        if function.parameters:

            lines.append(
                "**Parameters**"
            )

            lines.append("")

            for parameter in function.parameters:

                type_text = (
                    f" — `{parameter.annotation}`"
                    if parameter.annotation
                    else ""
                )

                default_text = (
                    f" — default: `{parameter.default}`"
                    if parameter.default
                    else ""
                )

                description = (
                    parameter.description
                    or "No description provided."
                )

                lines.append(
                    f"- `{parameter.name}`"
                    f"{type_text}"
                    f"{default_text}: "
                    f"{description}"
                )

            lines.append("")

        if function.return_annotation:

            lines.append(
                "**Returns:** "
                f"`{function.return_annotation}`"
            )

            lines.append("")

        return lines

    # =========================================================
    # PROJECT DOCUMENTATION
    # =========================================================

    def generate_project_markdown(
        self,
        modules: Sequence[
            ModuleDocumentation
        ],
        *,
        title: str = "RENIX Project Documentation",
    ) -> str:

        lines = [
            f"# {title}",
            "",
            "Generated automatically by RENIX.",
            "",
            "## Table of Contents",
            "",
        ]

        for module in modules:

            anchor = self._markdown_anchor(
                module.name
            )

            lines.append(
                f"- [{module.name}](#{anchor})"
            )

        lines.append("")

        for module in modules:

            lines.append(
                f"## `{module.name}`"
            )

            lines.append("")

            if module.docstring:

                lines.append(
                    self._clean_docstring(
                        module.docstring
                    )
                )

                lines.append("")

            if module.classes:

                lines.append(
                    "### Classes"
                )

                lines.append("")

                for class_doc in module.classes:

                    lines.append(
                        f"- `{class_doc.qualified_name}`"
                    )

                lines.append("")

            if module.functions:

                lines.append(
                    "### Functions"
                )

                lines.append("")

                for function in module.functions:

                    lines.append(
                        f"- `{function.qualified_name}`"
                    )

                lines.append("")

            lines.append(
                f"[Open full module documentation]"
                f"(modules/{module.name}.md)"
            )

            lines.append("")

        return "\n".join(
            lines
        ).rstrip() + "\n"

    # =========================================================
    # FILE GENERATION
    # =========================================================

    def generate_documentation(
        self,
        *,
        source_root: str | Path | None = None,
        output_directory: str | Path | None = None,
        title: str = "RENIX Project Documentation",
    ) -> DocumentationResult:

        root = (
            self.project_root
            if source_root is None
            else Path(source_root)
            .expanduser()
            .resolve()
        )

        output = (
            root / "docs"
            if output_directory is None
            else Path(output_directory)
            .expanduser()
            .resolve()
        )

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        modules_directory = (
            output / "modules"
        )

        modules_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        errors: list[str] = []

        modules = self.analyze_project(
            root
        )

        for module in modules:

            try:

                markdown = (
                    self.generate_module_markdown(
                        module
                    )
                )

                filename = (
                    self._safe_module_filename(
                        module.name
                    )
                    + ".md"
                )

                destination = (
                    modules_directory
                    / filename
                )

                destination.write_text(
                    markdown,
                    encoding="utf-8",
                )

            except Exception as exc:

                errors.append(
                    f"{module.name}: {exc}"
                )

        project_markdown = (
            self.generate_project_markdown(
                modules,
                title=title,
            )
        )

        index_file = (
            output / "README.md"
        )

        try:

            index_file.write_text(
                project_markdown,
                encoding="utf-8",
            )

        except Exception as exc:

            errors.append(
                f"README.md: {exc}"
            )

        function_count = sum(
            len(module.functions)
            + sum(
                len(class_doc.methods)
                for class_doc in module.classes
            )
            for module in modules
        )

        class_count = sum(
            len(module.classes)
            for module in modules
        )

        return DocumentationResult(
            success=not errors,
            output_file=str(index_file),
            modules_processed=len(modules),
            functions_documented=function_count,
            classes_documented=class_count,
            errors=errors,
        )

    # =========================================================
    # README UTILITIES
    # =========================================================

    def update_readme_section(
        self,
        readme_file: str | Path,
        section_title: str,
        content: str,
    ) -> Path:

        path = (
            Path(readme_file)
            .expanduser()
            .resolve()
        )

        if path.exists():
            original = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        else:
            original = ""

        heading = (
            f"## {section_title.strip()}"
        )

        pattern = re.compile(
            rf"(?ms)^"
            rf"{re.escape(heading)}"
            rf"\s*$.*?"
            rf"(?=^##\s|\Z)"
        )

        replacement = (
            f"{heading}\n\n"
            f"{content.strip()}\n\n"
        )

        if pattern.search(original):

            updated = pattern.sub(
                replacement,
                original,
                count=1,
            )

        else:

            separator = (
                "\n\n"
                if original.strip()
                else ""
            )

            updated = (
                original.rstrip()
                + separator
                + replacement
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            updated.rstrip() + "\n",
            encoding="utf-8",
        )

        return path

    # =========================================================
    # AST HELPERS
    # =========================================================

    @staticmethod
    def _extract_imports(
        node: ast.Import | ast.ImportFrom,
    ) -> list[str]:

        if isinstance(
            node,
            ast.Import,
        ):

            return [
                alias.name
                for alias in node.names
            ]

        module = (
            "." * node.level
            + (node.module or "")
        )

        return [
            f"{module}:{alias.name}"
            for alias in node.names
        ]

    @staticmethod
    def _extract_constants(
        node: ast.Assign,
    ) -> list[str]:

        result: list[str] = []

        for target in node.targets:

            if isinstance(
                target,
                ast.Name,
            ):

                if target.id.isupper():

                    result.append(
                        target.id
                    )

        return result

    @staticmethod
    def _extract_ann_constant(
        node: ast.AnnAssign,
    ) -> str | None:

        if (
            isinstance(
                node.target,
                ast.Name,
            )
            and node.target.id.isupper()
        ):

            return node.target.id

        return None

    @staticmethod
    def _annotation_to_string(
        node: ast.AST | None,
    ) -> str:

        if node is None:
            return ""

        try:
            return ast.unparse(
                node
            )
        except Exception:
            return "unknown"

    # =========================================================
    # PATH HELPERS
    # =========================================================

    def _module_name(
        self,
        path: Path,
    ) -> str:

        try:

            relative = path.relative_to(
                self.project_root
            )

        except ValueError:

            relative = path

        parts = list(
            relative.with_suffix("")
            .parts
        )

        if parts and parts[-1] == "__init__":
            parts = parts[:-1]

        if not parts:
            return path.stem

        return ".".join(
            parts
        )

    def _should_ignore(
        self,
        path: Path,
        root: Path,
    ) -> bool:

        try:

            relative = path.relative_to(
                root
            )

        except ValueError:

            return True

        return any(
            part in self.IGNORE_DIRECTORIES
            for part in relative.parts
        )

    @staticmethod
    def _safe_module_filename(
        module_name: str,
    ) -> str:

        return re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            module_name,
        )

    @staticmethod
    def _markdown_anchor(
        text: str,
    ) -> str:

        value = text.lower()

        value = re.sub(
            r"[^a-z0-9\s-]",
            "",
            value,
        )

        value = re.sub(
            r"\s+",
            "-",
            value,
        )

        return value.strip("-")

    @staticmethod
    def _clean_docstring(
        docstring: str,
    ) -> str:

        lines = [
            line.rstrip()
            for line in docstring.strip().splitlines()
        ]

        while lines and not lines[0]:
            lines.pop(0)

        while lines and not lines[-1]:
            lines.pop()

        return "\n".join(
            lines
        )


__all__ = [
    "DocumentationManager",
    "DocumentationResult",
    "ModuleDocumentation",
    "ClassDocumentation",
    "FunctionDocumentation",
    "ParameterDocumentation",
]


