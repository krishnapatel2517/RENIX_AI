"""
RENIX Analytics - Charts
========================

Chart and visualization utilities for RENIX analytics.

Matplotlib is loaded lazily so the analytics package can still be
imported when visualization dependencies are not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ChartEngine:
    """Create and save common analytics charts."""

    def __init__(
        self,
        output_directory: str | Path = "data/statistics/charts",
    ) -> None:
        self.output_directory = Path(output_directory)

    # ========================================================
    # DEPENDENCY
    # ========================================================

    @staticmethod
    def _get_matplotlib():
        """Load matplotlib lazily."""

        try:
            import matplotlib

            matplotlib.use("Agg")

            import matplotlib.pyplot as plt

            return plt

        except ImportError as exc:
            raise RuntimeError(
                "Chart support requires matplotlib. "
                "Install it with: pip install matplotlib"
            ) from exc

    # ========================================================
    # OUTPUT
    # ========================================================

    def _prepare_output(
        self,
        path: str | Path | None,
        default_name: str,
    ) -> Path:
        if path is None:
            output = (
                self.output_directory
                / default_name
            )
        else:
            output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output

    # ========================================================
    # LINE CHART
    # ========================================================

    def line(
        self,
        x: Sequence[Any],
        y: Sequence[float],
        *,
        title: str = "Line Chart",
        xlabel: str = "X",
        ylabel: str = "Y",
        output: str | Path | None = None,
        show_points: bool = True,
        grid: bool = True,
        figsize: tuple[float, float] = (10, 6),
    ) -> Path:
        """Create a line chart."""

        self._validate_lengths(x, y)

        plt = self._get_matplotlib()

        figure, axis = plt.subplots(
            figsize=figsize
        )

        axis.plot(
            x,
            y,
            marker="o" if show_points else None,
        )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

        if grid:
            axis.grid(True, alpha=0.3)

        figure.tight_layout()

        path = self._prepare_output(
            output,
            "line_chart.png",
        )

        figure.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        return path

    # ========================================================
    # BAR CHART
    # ========================================================

    def bar(
        self,
        categories: Sequence[Any],
        values: Sequence[float],
        *,
        title: str = "Bar Chart",
        xlabel: str = "Category",
        ylabel: str = "Value",
        output: str | Path | None = None,
        horizontal: bool = False,
        grid: bool = True,
        figsize: tuple[float, float] = (10, 6),
    ) -> Path:
        """Create a bar chart."""

        self._validate_lengths(
            categories,
            values,
        )

        plt = self._get_matplotlib()

        figure, axis = plt.subplots(
            figsize=figsize
        )

        if horizontal:
            axis.barh(
                categories,
                values,
            )
        else:
            axis.bar(
                categories,
                values,
            )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

        if grid:
            axis.grid(
                True,
                axis="x" if horizontal else "y",
                alpha=0.3,
            )

        figure.tight_layout()

        path = self._prepare_output(
            output,
            "bar_chart.png",
        )

        figure.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        return path

    # ========================================================
    # HISTOGRAM
    # ========================================================

    def histogram(
        self,
        values: Iterable[float],
        *,
        bins: int = 10,
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        output: str | Path | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> Path:
        """Create a histogram."""

        if bins <= 0:
            raise ValueError(
                "bins must be greater than 0"
            )

        data = list(values)

        if not data:
            raise ValueError(
                "Cannot create histogram from empty data."
            )

        plt = self._get_matplotlib()

        figure, axis = plt.subplots(
            figsize=figsize
        )

        axis.hist(
            data,
            bins=bins,
        )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

        axis.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        figure.tight_layout()

        path = self._prepare_output(
            output,
            "histogram.png",
        )

        figure.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        return path

    # ========================================================
    # SCATTER
    # ========================================================

    def scatter(
        self,
        x: Sequence[float],
        y: Sequence[float],
        *,
        title: str = "Scatter Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        output: str | Path | None = None,
        trendline: bool = False,
        figsize: tuple[float, float] = (10, 6),
    ) -> Path:
        """Create a scatter plot."""

        self._validate_lengths(x, y)

        if not x:
            raise ValueError(
                "Cannot create scatter plot from empty data."
            )

        plt = self._get_matplotlib()

        figure, axis = plt.subplots(
            figsize=figsize
        )

        axis.scatter(
            x,
            y,
        )

        if trendline and len(x) >= 2:
            try:
                import numpy as np

                x_array = np.asarray(
                    x,
                    dtype=float,
                )
                y_array = np.asarray(
                    y,
                    dtype=float,
                )

                coefficients = np.polyfit(
                    x_array,
                    y_array,
                    1,
                )

                trend = np.poly1d(coefficients)

                sorted_x = np.sort(x_array)

                axis.plot(
                    sorted_x,
                    trend(sorted_x),
                    linestyle="--",
                )

            except (ImportError, ValueError):
                pass

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()

        path = self._prepare_output(
            output,
            "scatter_plot.png",
        )

        figure.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        return path

    # ========================================================
    # PIE CHART
    # ========================================================

    def pie(
        self,
        labels: Sequence[Any],
        values: Sequence[float],
        *,
        title: str = "Pie Chart",
        output: str | Path | None = None,
        show_percentages: bool = True,
        figsize: tuple[float, float] = (8, 8),
    ) -> Path:
        """Create a pie chart."""

        self._validate_lengths(
            labels,
            values,
        )

        if not values:
            raise ValueError(
                "Cannot create pie chart from empty data."
            )

        if any(
            value < 0
            for value in values
        ):
            raise ValueError(
                "Pie chart values cannot be negative."
            )

        if sum(values) == 0:
            raise ValueError(
                "Pie chart values cannot all be zero."
            )

        plt = self._get_matplotlib()

        figure, axis = plt.subplots(
            figsize=figsize
        )

        axis.pie(
            values,
            labels=labels,
            autopct=(
                "%1.1f%%"
                if show_percentages
                else None
            ),
        )

        axis.set_title(title)

        figure.tight_layout()

        path = self._prepare_output(
            output,
            "pie_chart.png",
        )

        figure.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        return path

    # ========================================================
    # AREA CHART
    # ========================================================

    def area(
        self,
        x: Sequence[Any],
        y: Sequence[float],
        *,
        title: str = "Area Chart",
        xlabel: str = "X",
        ylabel: str = "Y",
        output: str | Path | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> Path:
        """Create an area chart."""

        self._validate_lengths(x, y)

        plt = self._get_matplotlib()

        figure, axis = plt.subplots(
            figsize=figsize
        )

        axis.fill_between(
            x,
            y,
            alpha=0.3,
        )

        axis.plot(
            x,
            y,
        )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()

        path = self._prepare_output(
            output,
            "area_chart.png",
        )

        figure.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        return path

    # ========================================================
    # MULTI-SERIES LINE
    # ========================================================

    def multi_line(
        self,
        x: Sequence[Any],
        series: Mapping[str, Sequence[float]],
        *,
        title: str = "Multi-Series Line Chart",
        xlabel: str = "X",
        ylabel: str = "Value",
        output: str | Path | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> Path:
        """Create a line chart with multiple series."""

        if not series:
            raise ValueError(
                "At least one series is required."
            )

        for name, values in series.items():
            self._validate_lengths(
                x,
                values,
            )

        plt = self._get_matplotlib()

        figure, axis = plt.subplots(
            figsize=figsize
        )

        for name, values in series.items():
            axis.plot(
                x,
                values,
                marker="o",
                label=str(name),
            )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

        axis.legend()

        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()

        path = self._prepare_output(
            output,
            "multi_line_chart.png",
        )

        figure.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        return path

    # ========================================================
    # BOXPLOT
    # ========================================================

    def boxplot(
        self,
        datasets: Mapping[str, Sequence[float]]
        | Sequence[Sequence[float]],
        *,
        title: str = "Box Plot",
        ylabel: str = "Value",
        output: str | Path | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> Path:
        """Create a box plot."""

        if not datasets:
            raise ValueError(
                "At least one dataset is required."
            )

        plt = self._get_matplotlib()

        figure, axis = plt.subplots(
            figsize=figsize
        )

        if isinstance(datasets, Mapping):
            labels = list(datasets.keys())
            values = list(datasets.values())

            axis.boxplot(
                values,
                labels=[
                    str(label)
                    for label in labels
                ],
            )

        else:
            axis.boxplot(datasets)

        axis.set_title(title)
        axis.set_ylabel(ylabel)

        axis.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        figure.tight_layout()

        path = self._prepare_output(
            output,
            "boxplot.png",
        )

        figure.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        return path

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    def save_figure(
        self,
        figure: Any,
        path: str | Path,
        *,
        dpi: int = 150,
    ) -> Path:
        """Save an externally created matplotlib figure."""

        output = self._prepare_output(
            path,
            "chart.png",
        )

        figure.savefig(
            output,
            dpi=dpi,
            bbox_inches="tight",
        )

        return output

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_lengths(
        first: Sequence[Any],
        second: Sequence[Any],
    ) -> None:
        if len(first) != len(second):
            raise ValueError(
                "Input sequences must have the same length."
            )

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            f"ChartEngine("
            f"output_directory={str(self.output_directory)!r})"
        )



