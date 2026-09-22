"""
RENIX - Point Utilities
=======================

Common 2D/3D point and vector utilities used throughout the RENIX
gesture and hand-tracking systems.

Designed to avoid duplicating coordinate mathematics across:
    - palm.py
    - hand_tracker.py
    - zoom.py
    - rotate.py
    - gesture_engine.py
    - gesture_mapper.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


# ============================================================================
# POINT
# ============================================================================


@dataclass(frozen=True)
class Point:
    """Immutable 3D point."""

    x: float
    y: float
    z: float = 0.0

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index: int) -> float:
        values = (self.x, self.y, self.z)
        return values[index]

    def to_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def to_2d(self) -> tuple[float, float]:
        return self.x, self.y

    def distance_to(self, other: "Point") -> float:
        return distance(self, other)

    def distance_2d_to(self, other: "Point") -> float:
        return distance_2d(self, other)

    def midpoint(self, other: "Point") -> "Point":
        return midpoint(self, other)

    def lerp(self, other: "Point", amount: float) -> "Point":
        return lerp(self, other, amount)

    def magnitude(self) -> float:
        return magnitude(self)

    def magnitude_2d(self) -> float:
        return magnitude_2d(self)

    def normalized(self) -> "Point":
        return normalize(self)

    def dot(self, other: "Point") -> float:
        return dot(self, other)

    def cross(self, other: "Point") -> "Point":
        return cross(self, other)

    def __add__(self, other: "Point") -> "Point":
        return Point(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: "Point") -> "Point":
        return Point(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(self, value: float) -> "Point":
        return Point(
            self.x * value,
            self.y * value,
            self.z * value,
        )

    def __rmul__(self, value: float) -> "Point":
        return self.__mul__(value)

    def __truediv__(self, value: float) -> "Point":
        if value == 0:
            raise ZeroDivisionError(
                "Cannot divide Point by zero"
            )

        return Point(
            self.x / value,
            self.y / value,
            self.z / value,
        )

    def __neg__(self) -> "Point":
        return Point(
            -self.x,
            -self.y,
            -self.z,
        )


# ============================================================================
# VECTOR ALIASES
# ============================================================================


Vector2 = Point
Vector3 = Point


# ============================================================================
# CONVERSION
# ============================================================================


def to_point(
    value: Point | Sequence[float] | Iterable[float],
) -> Point:
    """
    Convert a tuple/list/iterable into Point.
    """

    if isinstance(value, Point):
        return value

    values = list(value)

    if len(values) < 2:
        raise ValueError(
            "A point requires at least x and y coordinates."
        )

    return Point(
        float(values[0]),
        float(values[1]),
        float(values[2]) if len(values) >= 3 else 0.0,
    )


# ============================================================================
# DISTANCE
# ============================================================================


def distance(
    a: Point | Sequence[float],
    b: Point | Sequence[float],
) -> float:
    """Calculate 3D Euclidean distance."""

    a = to_point(a)
    b = to_point(b)

    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z

    return math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )


def distance_2d(
    a: Point | Sequence[float],
    b: Point | Sequence[float],
) -> float:
    """Calculate 2D Euclidean distance."""

    a = to_point(a)
    b = to_point(b)

    dx = a.x - b.x
    dy = a.y - b.y

    return math.sqrt(
        dx * dx +
        dy * dy
    )


# ============================================================================
# MIDPOINT
# ============================================================================


def midpoint(
    a: Point | Sequence[float],
    b: Point | Sequence[float],
) -> Point:
    """Return the midpoint between two points."""

    a = to_point(a)
    b = to_point(b)

    return Point(
        (a.x + b.x) / 2.0,
        (a.y + b.y) / 2.0,
        (a.z + b.z) / 2.0,
    )


# ============================================================================
# CENTROID
# ============================================================================


def centroid(
    points: Iterable[
        Point | Sequence[float]
    ],
) -> Point:
    """Calculate the centroid of multiple points."""

    normalized = [
        to_point(point)
        for point in points
    ]

    if not normalized:
        return Point(0.0, 0.0, 0.0)

    count = len(normalized)

    return Point(
        sum(p.x for p in normalized) / count,
        sum(p.y for p in normalized) / count,
        sum(p.z for p in normalized) / count,
    )


# ============================================================================
# LINEAR INTERPOLATION
# ============================================================================


def lerp(
    a: Point | Sequence[float],
    b: Point | Sequence[float],
    amount: float,
) -> Point:
    """
    Linear interpolation.

    amount:
        0.0 -> a
        1.0 -> b
    """

    a = to_point(a)
    b = to_point(b)

    amount = clamp(
        float(amount),
        0.0,
        1.0,
    )

    return Point(
        a.x + (b.x - a.x) * amount,
        a.y + (b.y - a.y) * amount,
        a.z + (b.z - a.z) * amount,
    )


# ============================================================================
# MAGNITUDE
# ============================================================================


def magnitude(
    point: Point | Sequence[float],
) -> float:
    """Return 3D vector magnitude."""

    point = to_point(point)

    return math.sqrt(
        point.x ** 2 +
        point.y ** 2 +
        point.z ** 2
    )


def magnitude_2d(
    point: Point | Sequence[float],
) -> float:
    """Return 2D vector magnitude."""

    point = to_point(point)

    return math.sqrt(
        point.x ** 2 +
        point.y ** 2
    )


# ============================================================================
# NORMALIZATION
# ============================================================================


def normalize(
    point: Point | Sequence[float],
) -> Point:
    """Return normalized 3D vector."""

    point = to_point(point)

    length = magnitude(point)

    if length <= 1e-12:
        return Point(
            0.0,
            0.0,
            0.0,
        )

    return point / length


# ============================================================================
# DOT PRODUCT
# ============================================================================


def dot(
    a: Point | Sequence[float],
    b: Point | Sequence[float],
) -> float:
    """Calculate dot product."""

    a = to_point(a)
    b = to_point(b)

    return (
        a.x * b.x +
        a.y * b.y +
        a.z * b.z
    )


# ============================================================================
# CROSS PRODUCT
# ============================================================================


def cross(
    a: Point | Sequence[float],
    b: Point | Sequence[float],
) -> Point:
    """Calculate 3D cross product."""

    a = to_point(a)
    b = to_point(b)

    return Point(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


# ============================================================================
# ANGLE
# ============================================================================


def angle_between(
    a: Point | Sequence[float],
    b: Point | Sequence[float],
) -> float:
    """
    Calculate angle between vectors in degrees.
    """

    a = to_point(a)
    b = to_point(b)

    magnitude_a = magnitude(a)
    magnitude_b = magnitude(b)

    if (
        magnitude_a <= 1e-12
        or magnitude_b <= 1e-12
    ):
        return 0.0

    cosine = dot(a, b) / (
        magnitude_a * magnitude_b
    )

    cosine = clamp(
        cosine,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(cosine)
    )


def angle_2d(
    a: Point | Sequence[float],
    b: Point | Sequence[float],
) -> float:
    """
    Calculate the direction angle of vector a -> b
    in degrees.
    """

    a = to_point(a)
    b = to_point(b)

    return math.degrees(
        math.atan2(
            b.y - a.y,
            b.x - a.x,
        )
    )


# ============================================================================
# ANGLE NORMALIZATION
# ============================================================================


def normalize_angle(
    angle: float,
) -> float:
    """
    Normalize angle into [-180, 180).
    """

    result = (
        float(angle) + 180.0
    ) % 360.0 - 180.0

    return result


def angle_difference(
    current: float,
    previous: float,
) -> float:
    """Shortest signed angular difference."""

    return normalize_angle(
        float(current)
        - float(previous)
    )


# ============================================================================
# ROTATION
# ============================================================================


def rotate_2d(
    point: Point | Sequence[float],
    angle_degrees: float,
    origin: Point | Sequence[float] = Point(
        0.0,
        0.0,
        0.0,
    ),
) -> Point:
    """Rotate a point around an origin in the XY plane."""

    point = to_point(point)
    origin = to_point(origin)

    radians = math.radians(
        angle_degrees
    )

    cosine = math.cos(radians)
    sine = math.sin(radians)

    x = point.x - origin.x
    y = point.y - origin.y

    rotated_x = (
        x * cosine
        - y * sine
    )

    rotated_y = (
        x * sine
        + y * cosine
    )

    return Point(
        rotated_x + origin.x,
        rotated_y + origin.y,
        point.z,
    )


# ============================================================================
# BOUNDING BOX
# ============================================================================


def bounding_box(
    points: Iterable[
        Point | Sequence[float]
    ],
) -> tuple[Point, Point]:
    """
    Return minimum and maximum points.
    """

    normalized = [
        to_point(point)
        for point in points
    ]

    if not normalized:
        zero = Point(0.0, 0.0, 0.0)
        return zero, zero

    minimum = Point(
        min(p.x for p in normalized),
        min(p.y for p in normalized),
        min(p.z for p in normalized),
    )

    maximum = Point(
        max(p.x for p in normalized),
        max(p.y for p in normalized),
        max(p.z for p in normalized),
    )

    return minimum, maximum


# ============================================================================
# CLAMP
# ============================================================================


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Clamp value between minimum and maximum."""

    if minimum > maximum:
        minimum, maximum = (
            maximum,
            minimum,
        )

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


# ============================================================================
# REMAP
# ============================================================================


def remap(
    value: float,
    input_min: float,
    input_max: float,
    output_min: float,
    output_max: float,
    *,
    clamp_result: bool = False,
) -> float:
    """
    Remap a value from one range into another.
    """

    input_range = (
        input_max - input_min
    )

    if abs(input_range) <= 1e-12:
        return output_min

    normalized = (
        value - input_min
    ) / input_range

    if clamp_result:
        normalized = clamp(
            normalized,
            0.0,
            1.0,
        )

    return (
        output_min
        + normalized
        * (
            output_max
            - output_min
        )
    )


# ============================================================================
# POINT IN RECTANGLE
# ============================================================================


def point_in_rect(
    point: Point | Sequence[float],
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> bool:
    """Check whether a point lies inside a rectangle."""

    point = to_point(point)

    return (
        left <= point.x <= right
        and top <= point.y <= bottom
    )


# ============================================================================
# DISTANCE FROM LINE
# ============================================================================


def distance_from_line(
    point: Point | Sequence[float],
    line_start: Point | Sequence[float],
    line_end: Point | Sequence[float],
) -> float:
    """Calculate 2D perpendicular distance from a point to a line."""

    point = to_point(point)
    line_start = to_point(line_start)
    line_end = to_point(line_end)

    dx = (
        line_end.x
        - line_start.x
    )

    dy = (
        line_end.y
        - line_start.y
    )

    length = math.sqrt(
        dx * dx +
        dy * dy
    )

    if length <= 1e-12:
        return distance_2d(
            point,
            line_start,
        )

    numerator = abs(
        dy * point.x
        - dx * point.y
        + line_end.x * line_start.y
        - line_end.y * line_start.x
    )

    return numerator / length


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "Point",
    "Vector2",
    "Vector3",
    "to_point",
    "distance",
    "distance_2d",
    "midpoint",
    "centroid",
    "lerp",
    "magnitude",
    "magnitude_2d",
    "normalize",
    "dot",
    "cross",
    "angle_between",
    "angle_2d",
    "normalize_angle",
    "angle_difference",
    "rotate_2d",
    "bounding_box",
    "clamp",
    "remap",
    "point_in_rect",
    "distance_from_line",
]


