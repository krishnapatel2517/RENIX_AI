"""
RENIX Analytics - Statistics Engine
===================================

Statistical analysis utilities for RENIX.

Provides:
- Descriptive statistics
- Percentiles and quartiles
- Variance and standard deviation
- Correlation
- Covariance
- Z-scores
- Outlier detection
- Frequency analysis
- Numeric summaries
"""

from __future__ import annotations

import math
import statistics as stats
from collections import Counter
from typing import Any, Iterable, Mapping


class StatisticsEngine:
    """General-purpose statistical analysis engine."""

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _numbers(values: Iterable[Any]) -> list[float]:
        """Convert an iterable into usable numeric values."""

        result: list[float] = []

        for value in values:
            if value is None or value == "":
                continue

            if isinstance(value, bool):
                continue

            try:
                number = float(value)

                if math.isfinite(number):
                    result.append(number)

            except (TypeError, ValueError):
                continue

        return result

    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    def count(self, values: Iterable[Any]) -> int:
        return len(self._numbers(values))

    def sum(self, values: Iterable[Any]) -> float:
        return sum(self._numbers(values))

    def mean(self, values: Iterable[Any]) -> float | None:
        numbers = self._numbers(values)

        if not numbers:
            return None

        return stats.mean(numbers)

    def median(self, values: Iterable[Any]) -> float | None:
        numbers = self._numbers(values)

        if not numbers:
            return None

        return stats.median(numbers)

    def mode(self, values: Iterable[Any]) -> float | None:
        numbers = self._numbers(values)

        if not numbers:
            return None

        try:
            return stats.mode(numbers)
        except stats.StatisticsError:
            return None

    def minimum(self, values: Iterable[Any]) -> float | None:
        numbers = self._numbers(values)

        return min(numbers) if numbers else None

    def maximum(self, values: Iterable[Any]) -> float | None:
        numbers = self._numbers(values)

        return max(numbers) if numbers else None

    def range(self, values: Iterable[Any]) -> float | None:
        numbers = self._numbers(values)

        if not numbers:
            return None

        return max(numbers) - min(numbers)

    # ========================================================
    # VARIANCE / DEVIATION
    # ========================================================

    def variance(
        self,
        values: Iterable[Any],
        *,
        sample: bool = True,
    ) -> float | None:
        """Calculate variance."""

        numbers = self._numbers(values)

        if len(numbers) < (2 if sample else 1):
            return None

        return (
            stats.variance(numbers)
            if sample
            else stats.pvariance(numbers)
        )

    def standard_deviation(
        self,
        values: Iterable[Any],
        *,
        sample: bool = True,
    ) -> float | None:
        """Calculate standard deviation."""

        numbers = self._numbers(values)

        if len(numbers) < (2 if sample else 1):
            return None

        return (
            stats.stdev(numbers)
            if sample
            else stats.pstdev(numbers)
        )

    # ========================================================
    # PERCENTILES
    # ========================================================

    def percentile(
        self,
        values: Iterable[Any],
        percentile: float,
    ) -> float | None:
        """
        Calculate a percentile.

        percentile must be between 0 and 100.
        """

        if not 0 <= percentile <= 100:
            raise ValueError(
                "percentile must be between 0 and 100"
            )

        numbers = sorted(self._numbers(values))

        if not numbers:
            return None

        if len(numbers) == 1:
            return numbers[0]

        position = (
            (len(numbers) - 1)
            * percentile
            / 100
        )

        lower = math.floor(position)
        upper = math.ceil(position)

        if lower == upper:
            return numbers[lower]

        weight = position - lower

        return (
            numbers[lower]
            + (numbers[upper] - numbers[lower])
            * weight
        )

    def quartiles(
        self,
        values: Iterable[Any],
    ) -> dict[str, float | None]:
        """Return Q1, median, and Q3."""

        numbers = self._numbers(values)

        if not numbers:
            return {
                "q1": None,
                "median": None,
                "q3": None,
            }

        return {
            "q1": self.percentile(numbers, 25),
            "median": self.percentile(numbers, 50),
            "q3": self.percentile(numbers, 75),
        }

    def interquartile_range(
        self,
        values: Iterable[Any],
    ) -> float | None:
        """Return Q3 - Q1."""

        quartiles = self.quartiles(values)

        q1 = quartiles["q1"]
        q3 = quartiles["q3"]

        if q1 is None or q3 is None:
            return None

        return q3 - q1

    # ========================================================
    # FREQUENCY
    # ========================================================

    def frequency(
        self,
        values: Iterable[Any],
    ) -> dict[Any, int]:
        """Count occurrences of values."""

        return dict(Counter(values))

    def relative_frequency(
        self,
        values: Iterable[Any],
    ) -> dict[Any, float]:
        """Return frequencies as proportions."""

        counter = Counter(values)
        total = sum(counter.values())

        if total == 0:
            return {}

        return {
            key: count / total
            for key, count in counter.items()
        }

    def top_values(
        self,
        values: Iterable[Any],
        limit: int = 10,
    ) -> list[tuple[Any, int]]:
        """Return the most frequent values."""

        if limit < 1:
            return []

        return Counter(values).most_common(limit)

    # ========================================================
    # CORRELATION
    # ========================================================

    def covariance(
        self,
        x: Iterable[Any],
        y: Iterable[Any],
        *,
        sample: bool = True,
    ) -> float | None:
        """Calculate covariance between two datasets."""

        x_values = self._numbers(x)
        y_values = self._numbers(y)

        if len(x_values) != len(y_values):
            raise ValueError(
                "x and y must contain the same number "
                "of numeric values."
            )

        minimum_length = 2 if sample else 1

        if len(x_values) < minimum_length:
            return None

        x_mean = stats.mean(x_values)
        y_mean = stats.mean(y_values)

        total = sum(
            (a - x_mean) * (b - y_mean)
            for a, b in zip(
                x_values,
                y_values,
            )
        )

        denominator = (
            len(x_values) - 1
            if sample
            else len(x_values)
        )

        return total / denominator

    def correlation(
        self,
        x: Iterable[Any],
        y: Iterable[Any],
    ) -> float | None:
        """Calculate Pearson correlation."""

        x_values = self._numbers(x)
        y_values = self._numbers(y)

        if len(x_values) != len(y_values):
            raise ValueError(
                "x and y must contain the same number "
                "of numeric values."
            )

        if len(x_values) < 2:
            return None

        x_mean = stats.mean(x_values)
        y_mean = stats.mean(y_values)

        numerator = sum(
            (a - x_mean) * (b - y_mean)
            for a, b in zip(
                x_values,
                y_values,
            )
        )

        x_sum = sum(
            (a - x_mean) ** 2
            for a in x_values
        )

        y_sum = sum(
            (b - y_mean) ** 2
            for b in y_values
        )

        denominator = math.sqrt(
            x_sum * y_sum
        )

        if denominator == 0:
            return None

        return numerator / denominator

    # ========================================================
    # Z-SCORES
    # ========================================================

    def z_scores(
        self,
        values: Iterable[Any],
        *,
        sample: bool = False,
    ) -> list[float]:
        """Calculate z-scores."""

        numbers = self._numbers(values)

        if not numbers:
            return []

        deviation = self.standard_deviation(
            numbers,
            sample=sample,
        )

        if deviation is None or deviation == 0:
            return [0.0 for _ in numbers]

        average = stats.mean(numbers)

        return [
            (value - average) / deviation
            for value in numbers
        ]

    # ========================================================
    # OUTLIERS
    # ========================================================

    def outliers_iqr(
        self,
        values: Iterable[Any],
        *,
        multiplier: float = 1.5,
    ) -> list[float]:
        """Detect outliers using the IQR method."""

        if multiplier < 0:
            raise ValueError(
                "multiplier must be >= 0"
            )

        numbers = self._numbers(values)

        if len(numbers) < 2:
            return []

        quartiles = self.quartiles(numbers)

        q1 = quartiles["q1"]
        q3 = quartiles["q3"]

        if q1 is None or q3 is None:
            return []

        iqr = q3 - q1

        lower_bound = (
            q1 - multiplier * iqr
        )

        upper_bound = (
            q3 + multiplier * iqr
        )

        return [
            value
            for value in numbers
            if value < lower_bound
            or value > upper_bound
        ]

    def outliers_zscore(
        self,
        values: Iterable[Any],
        *,
        threshold: float = 3.0,
    ) -> list[float]:
        """Detect outliers using z-scores."""

        if threshold < 0:
            raise ValueError(
                "threshold must be >= 0"
            )

        numbers = self._numbers(values)
        scores = self.z_scores(numbers)

        return [
            value
            for value, score in zip(
                numbers,
                scores,
            )
            if abs(score) > threshold
        ]

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize(
        self,
        values: Iterable[Any],
    ) -> list[float]:
        """
        Min-max normalization to [0, 1].
        """

        numbers = self._numbers(values)

        if not numbers:
            return []

        minimum = min(numbers)
        maximum = max(numbers)

        if minimum == maximum:
            return [0.0 for _ in numbers]

        return [
            (value - minimum)
            / (maximum - minimum)
            for value in numbers
        ]

    def standardize(
        self,
        values: Iterable[Any],
    ) -> list[float]:
        """Standardize values using population z-scores."""

        return self.z_scores(
            values,
            sample=False,
        )

    # ========================================================
    # RANKING
    # ========================================================

    def rank(
        self,
        values: Iterable[Any],
        *,
        descending: bool = False,
    ) -> list[int]:
        """Return ordinal ranks for values."""

        numbers = self._numbers(values)

        indexed = list(
            enumerate(numbers)
        )

        indexed.sort(
            key=lambda item: item[1],
            reverse=descending,
        )

        ranks = [0] * len(numbers)

        for position, (index, _) in enumerate(
            indexed,
            start=1,
        ):
            ranks[index] = position

        return ranks

    # ========================================================
    # SUMMARY
    # ========================================================

    def describe(
        self,
        values: Iterable[Any],
    ) -> dict[str, Any]:
        """Generate a complete descriptive summary."""

        numbers = self._numbers(values)

        if not numbers:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": None,
                "median": None,
                "mode": None,
                "minimum": None,
                "maximum": None,
                "range": None,
                "variance": None,
                "standard_deviation": None,
                "q1": None,
                "q3": None,
                "iqr": None,
            }

        quartiles = self.quartiles(numbers)

        return {
            "count": len(numbers),
            "sum": sum(numbers),
            "mean": stats.mean(numbers),
            "median": stats.median(numbers),
            "mode": self.mode(numbers),
            "minimum": min(numbers),
            "maximum": max(numbers),
            "range": max(numbers) - min(numbers),
            "variance": self.variance(
                numbers,
                sample=True,
            ),
            "standard_deviation": self.standard_deviation(
                numbers,
                sample=True,
            ),
            "q1": quartiles["q1"],
            "q3": quartiles["q3"],
            "iqr": self.interquartile_range(numbers),
        }

    # ========================================================
    # COLUMN ANALYSIS
    # ========================================================

    def analyze_column(
        self,
        records: Iterable[Mapping[str, Any]],
        column: str,
    ) -> dict[str, Any]:
        """Analyze a numeric column from record data."""

        values = [
            record.get(column)
            for record in records
            if isinstance(record, Mapping)
        ]

        return self.describe(values)

    # ========================================================
    # MULTI-COLUMN ANALYSIS
    # ========================================================

    def analyze_dataset(
        self,
        records: Iterable[Mapping[str, Any]],
        columns: Iterable[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Analyze multiple numeric columns."""

        records_list = list(records)

        if columns is None:
            discovered: list[str] = []

            for record in records_list:
                if not isinstance(record, Mapping):
                    continue

                for key in record:
                    if key not in discovered:
                        discovered.append(key)

            columns = discovered

        result: dict[str, dict[str, Any]] = {}

        for column in columns:
            result[str(column)] = self.analyze_column(
                records_list,
                str(column),
            )

        return result

    # ========================================================
    # PERCENTILE RANK
    # ========================================================

    def percentile_rank(
        self,
        values: Iterable[Any],
        value: float,
    ) -> float | None:
        """Return the percentile rank of a value."""

        numbers = self._numbers(values)

        if not numbers:
            return None

        below_or_equal = sum(
            1
            for number in numbers
            if number <= value
        )

        return (
            below_or_equal
            / len(numbers)
            * 100
        )

    # ========================================================
    # MOVING AVERAGE
    # ========================================================

    def moving_average(
        self,
        values: Iterable[Any],
        window: int = 3,
    ) -> list[float]:
        """Calculate a simple moving average."""

        if window <= 0:
            raise ValueError(
                "window must be greater than 0"
            )

        numbers = self._numbers(values)

        if len(numbers) < window:
            return []

        return [
            stats.mean(
                numbers[index - window + 1:index + 1]
            )
            for index in range(
                window - 1,
                len(numbers),
            )
        ]

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return "StatisticsEngine()"



