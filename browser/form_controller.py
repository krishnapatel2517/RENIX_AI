"""
RENIX Browser Form Controller
=============================

Controls HTML forms through the active browser.

Responsibilities:
    - Inspect forms
    - Read form fields
    - Fill inputs
    - Select options
    - Toggle checkboxes
    - Submit forms
    - Clear form fields
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FormController:
    """Control forms and form fields in browser pages."""

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
    # FORM DISCOVERY
    # ========================================================

    def get_forms(
        self,
        *,
        page_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all forms on the current page."""

        page = self._get_page(
            page_id
        )

        script = """
        () => Array.from(
            document.querySelectorAll("form")
        ).map((form, index) => ({
            index,
            id: form.id || "",
            name: form.getAttribute("name") || "",
            action: form.action || "",
            method: (
                form.method || "get"
            ).toLowerCase(),
            enctype: form.enctype || "",
            fields: Array.from(
                form.elements
            ).map((element) => ({
                tag: element.tagName.toLowerCase(),
                type: element.type || "",
                name: element.name || "",
                id: element.id || "",
                value: element.value || "",
                placeholder:
                    element.getAttribute("placeholder") || "",
                required: !!element.required,
                disabled: !!element.disabled
            }))
        }))
        """

        return self._evaluate(
            page,
            script,
        )

    def get_form(
        self,
        *,
        form_selector: str | None = None,
        form_index: int | None = None,
        page_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return one form."""

        forms = self.get_forms(
            page_id=page_id
        )

        if form_selector:
            page = self._get_page(
                page_id
            )

            script = """
            (selector) => {
                const form =
                    document.querySelector(selector);

                if (!form) {
                    return null;
                }

                return {
                    id: form.id || "",
                    name:
                        form.getAttribute("name") || "",
                    action: form.action || "",
                    method:
                        (
                            form.method || "get"
                        ).toLowerCase(),
                    fields: Array.from(
                        form.elements
                    ).map((element) => ({
                        tag:
                            element.tagName.toLowerCase(),
                        type: element.type || "",
                        name: element.name || "",
                        id: element.id || "",
                        value: element.value || "",
                        required:
                            !!element.required,
                        disabled:
                            !!element.disabled
                    }))
                };
            }
            """

            return self._evaluate(
                page,
                script,
                form_selector,
                single=True,
            )

        if form_index is None:
            form_index = 0

        if (
            form_index < 0
            or form_index >= len(forms)
        ):
            return None

        return forms[
            form_index
        ]

    # ========================================================
    # FIELD DISCOVERY
    # ========================================================

    def get_fields(
        self,
        *,
        form_selector: str | None = None,
        form_index: int | None = None,
        page_id: str | None = None,
    ) -> list[dict[str, Any]]:

        form = self.get_form(
            form_selector=form_selector,
            form_index=form_index,
            page_id=page_id,
        )

        if not form:
            return []

        return form.get(
            "fields",
            [],
        )

    # ========================================================
    # FILL
    # ========================================================

    def fill(
        self,
        selector: str,
        value: Any,
        *,
        page_id: str | None = None,
    ) -> bool:
        """Fill a text-like form field."""

        page = self._get_page(
            page_id
        )

        selector = selector.strip()

        if not selector:
            raise ValueError(
                "Field selector cannot be empty."
            )

        value = "" if value is None else str(
            value
        )

        locator = self._locator(
            page,
            selector,
        )

        fill = getattr(
            locator,
            "fill",
            None,
        )

        if callable(fill):

            fill(
                value
            )

            return True

        # JavaScript fallback.
        script = """
        (args) => {
            const element =
                document.querySelector(args.selector);

            if (!element) {
                return false;
            }

            element.focus();
            element.value = args.value;

            element.dispatchEvent(
                new Event(
                    "input",
                    { bubbles: true }
                )
            );

            element.dispatchEvent(
                new Event(
                    "change",
                    { bubbles: true }
                )
            );

            return true;
        }
        """

        return bool(
            self._evaluate(
                page,
                script,
                {
                    "selector": selector,
                    "value": value,
                },
                single=True,
            )
        )

    # ========================================================
    # TYPE
    # ========================================================

    def type_text(
        self,
        selector: str,
        value: str,
        *,
        page_id: str | None = None,
        delay: float | None = None,
    ) -> bool:
        """Type text into a field."""

        page = self._get_page(
            page_id
        )

        locator = self._locator(
            page,
            selector,
        )

        fill = getattr(
            locator,
            "fill",
            None,
        )

        if callable(fill):
            fill(
                str(value)
            )
            return True

        type_method = getattr(
            locator,
            "type",
            None,
        )

        if callable(type_method):

            if delay is not None:
                try:
                    type_method(
                        str(value),
                        delay=delay,
                    )
                except TypeError:
                    type_method(
                        str(value)
                    )
            else:
                type_method(
                    str(value)
                )

            return True

        return self.fill(
            selector,
            value,
            page_id=page_id,
        )

    # ========================================================
    # SELECT
    # ========================================================

    def select(
        self,
        selector: str,
        value: str | list[str],
        *,
        page_id: str | None = None,
    ) -> bool:
        """Select one or more options from a select element."""

        page = self._get_page(
            page_id
        )

        locator = self._locator(
            page,
            selector,
        )

        select_option = getattr(
            locator,
            "select_option",
            None,
        )

        if callable(select_option):

            select_option(
                value
            )

            return True

        script = """
        (args) => {
            const select =
                document.querySelector(args.selector);

            if (!select) {
                return false;
            }

            const values =
                Array.isArray(args.value)
                    ? args.value
                    : [args.value];

            for (
                const option
                of select.options
            ) {
                option.selected =
                    values.includes(
                        option.value
                    ) ||
                    values.includes(
                        option.text
                    );
            }

            select.dispatchEvent(
                new Event(
                    "change",
                    { bubbles: true }
                )
            );

            return true;
        }
        """

        return bool(
            self._evaluate(
                page,
                script,
                {
                    "selector": selector,
                    "value": value,
                },
                single=True,
            )
        )

    # ========================================================
    # CHECKBOX
    # ========================================================

    def check(
        self,
        selector: str,
        *,
        page_id: str | None = None,
    ) -> bool:
        """Check a checkbox."""

        return self._set_checkbox(
            selector,
            True,
            page_id=page_id,
        )

    def uncheck(
        self,
        selector: str,
        *,
        page_id: str | None = None,
    ) -> bool:
        """Uncheck a checkbox."""

        return self._set_checkbox(
            selector,
            False,
            page_id=page_id,
        )

    def _set_checkbox(
        self,
        selector: str,
        checked: bool,
        *,
        page_id: str | None = None,
    ) -> bool:

        page = self._get_page(
            page_id
        )

        locator = self._locator(
            page,
            selector,
        )

        method_name = (
            "check"
            if checked
            else "uncheck"
        )

        method = getattr(
            locator,
            method_name,
            None,
        )

        if callable(method):

            method()

            return True

        script = """
        (args) => {
            const element =
                document.querySelector(args.selector);

            if (!element) {
                return false;
            }

            if (
                element.type !== "checkbox" &&
                element.type !== "radio"
            ) {
                return false;
            }

            element.checked = args.checked;

            element.dispatchEvent(
                new Event(
                    "change",
                    { bubbles: true }
                )
            );

            return true;
        }
        """

        return bool(
            self._evaluate(
                page,
                script,
                {
                    "selector": selector,
                    "checked": checked,
                },
                single=True,
            )
        )

    # ========================================================
    # RADIO
    # ========================================================

    def choose_radio(
        self,
        selector: str,
        *,
        page_id: str | None = None,
    ) -> bool:

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

        if callable(click):

            click()

            return True

        return False

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(
        self,
        selector: str,
        *,
        page_id: str | None = None,
    ) -> bool:
        """Clear a form field."""

        return self.fill(
            selector,
            "",
            page_id=page_id,
        )

    # ========================================================
    # SUBMIT
    # ========================================================

    def submit(
        self,
        *,
        form_selector: str | None = None,
        form_index: int | None = None,
        submit_selector: str | None = None,
        page_id: str | None = None,
    ) -> bool:
        """Submit a form."""

        page = self._get_page(
            page_id
        )

        if submit_selector:

            locator = self._locator(
                page,
                submit_selector,
            )

            click = getattr(
                locator,
                "click",
                None,
            )

            if callable(click):

                click()

                return True

        if form_selector:

            locator = self._locator(
                page,
                form_selector,
            )

            submit = getattr(
                locator,
                "submit",
                None,
            )

            if callable(submit):

                submit()

                return True

        if form_index is None:
            form_index = 0

        script = """
        (index) => {
            const forms =
                document.querySelectorAll("form");

            const form = forms[index];

            if (!form) {
                return false;
            }

            if (
                typeof form.requestSubmit ===
                "function"
            ) {
                form.requestSubmit();
            } else {
                form.submit();
            }

            return true;
        }
        """

        return bool(
            self._evaluate(
                page,
                script,
                form_index,
                single=True,
            )
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        *,
        form_selector: str | None = None,
        form_index: int | None = None,
        page_id: str | None = None,
    ) -> bool:

        page = self._get_page(
            page_id
        )

        if form_selector:

            locator = self._locator(
                page,
                form_selector,
            )

            reset = getattr(
                locator,
                "reset",
                None,
            )

            if callable(reset):

                reset()

                return True

        index = (
            form_index
            if form_index is not None
            else 0
        )

        script = """
        (index) => {
            const forms =
                document.querySelectorAll("form");

            const form = forms[index];

            if (!form) {
                return false;
            }

            form.reset();

            return true;
        }
        """

        return bool(
            self._evaluate(
                page,
                script,
                index,
                single=True,
            )
        )

    # ========================================================
    # FIELD VALUE
    # ========================================================

    def get_value(
        self,
        selector: str,
        *,
        page_id: str | None = None,
    ) -> Any:

        page = self._get_page(
            page_id
        )

        locator = self._locator(
            page,
            selector,
        )

        input_value = getattr(
            locator,
            "input_value",
            None,
        )

        if callable(input_value):

            return input_value()

        script = """
        (selector) => {
            const element =
                document.querySelector(selector);

            if (!element) {
                return null;
            }

            return element.value ?? null;
        }
        """

        return self._evaluate(
            page,
            script,
            selector,
            single=True,
        )

    # ========================================================
    # FIELD STATE
    # ========================================================

    def is_checked(
        self,
        selector: str,
        *,
        page_id: str | None = None,
    ) -> bool:

        page = self._get_page(
            page_id
        )

        locator = self._locator(
            page,
            selector,
        )

        is_checked = getattr(
            locator,
            "is_checked",
            None,
        )

        if callable(is_checked):
            return bool(
                is_checked()
            )

        script = """
        (selector) => {
            const element =
                document.querySelector(selector);

            return !!(
                element &&
                element.checked
            );
        }
        """

        return bool(
            self._evaluate(
                page,
                script,
                selector,
                single=True,
            )
        )

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
        *,
        single: bool = False,
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

        try:

            if argument is None:
                result = evaluator(
                    script
                )
            else:
                result = evaluator(
                    script,
                    argument,
                )

        except TypeError:

            # Some browser backends only accept
            # the JavaScript function itself.
            if argument is None:
                result = evaluator(
                    script
                )
            else:
                result = evaluator(
                    f"""
                    () => {{
                        const args = {repr(argument)};
                        return ({script})(args);
                    }}
                    """
                )

        if single:
            return result

        if isinstance(
            result,
            list,
        ):
            return result

        return []


__all__ = [
    "FormController",
]


