"""
RENIX Education — Mathematics Subject Engine

Provides:
- Mathematics topic/chapter management
- Question generation
- Answer checking
- Difficulty classification
- Topic recommendations
- Practice sessions
- Formula storage
- Progress analysis
- Hint generation
- Step-by-step solution support
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class MathQuestion:
    question_id: str
    topic: str
    question: str
    answer: Any
    difficulty: str = "medium"
    explanation: str = ""
    hint: str = ""
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "topic": self.topic,
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "explanation": self.explanation,
            "hint": self.hint,
            "options": self.options,
        }


@dataclass
class MathTopic:
    name: str
    description: str
    formulas: list[str] = field(
        default_factory=list
    )
    difficulty: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "formulas": list(self.formulas),
            "difficulty": self.difficulty,
        }


# ============================================================
# MATHEMATICS ENGINE
# ============================================================


class MathematicsEngine:
    """
    RENIX mathematics subject engine.

    Example:

        maths = MathematicsEngine()

        question = maths.generate_question(
            "quadratic_equations"
        )

        result = maths.check_answer(
            question,
            "25"
        )
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:

        self.random = random.Random(seed)

        self.topics: dict[
            str,
            MathTopic,
        ] = {}

        self.question_counter = 0

        self._register_default_topics()

    # ========================================================
    # TOPICS
    # ========================================================

    def _register_default_topics(
        self,
    ) -> None:

        topics = [
            MathTopic(
                "real_numbers",
                "Euclid's division algorithm, HCF, LCM and irrational numbers.",
                [
                    "a = bq + r",
                    "HCF(a,b) × LCM(a,b) = a × b",
                ],
                "medium",
            ),
            MathTopic(
                "polynomials",
                "Zeros of polynomials and relationships between coefficients and zeros.",
                [
                    "For ax²+bx+c, sum of zeros = -b/a",
                    "For ax²+bx+c, product of zeros = c/a",
                ],
                "medium",
            ),
            MathTopic(
                "pair_of_linear_equations",
                "Solving two-variable linear equations.",
                [
                    "a₁x+b₁y+c₁=0",
                    "a₂x+b₂y+c₂=0",
                ],
                "medium",
            ),
            MathTopic(
                "quadratic_equations",
                "Quadratic equations, roots and discriminant.",
                [
                    "ax²+bx+c=0",
                    "D=b²-4ac",
                    "x=(-b±√D)/(2a)",
                ],
                "hard",
            ),
            MathTopic(
                "arithmetic_progressions",
                "Arithmetic sequences and their sums.",
                [
                    "aₙ=a+(n-1)d",
                    "Sₙ=n/2[2a+(n-1)d]",
                ],
                "medium",
            ),
            MathTopic(
                "triangles",
                "Similarity, proportionality and Pythagoras theorem.",
                [
                    "a²+b²=c²",
                ],
                "medium",
            ),
            MathTopic(
                "coordinate_geometry",
                "Distance, section formula and area of triangles.",
                [
                    "d=√((x₂-x₁)²+(y₂-y₁)²)",
                    "Midpoint=((x₁+x₂)/2,(y₁+y₂)/2)",
                ],
                "medium",
            ),
            MathTopic(
                "trigonometry",
                "Trigonometric ratios and identities.",
                [
                    "sin θ = perpendicular/hypotenuse",
                    "cos θ = base/hypotenuse",
                    "tan θ = perpendicular/base",
                    "sin²θ+cos²θ=1",
                ],
                "hard",
            ),
            MathTopic(
                "applications_of_trigonometry",
                "Heights and distances using trigonometry.",
                [
                    "tan θ = height/base",
                ],
                "hard",
            ),
            MathTopic(
                "circles",
                "Tangents and properties of circles.",
                [
                    "Radius ⟂ tangent",
                ],
                "medium",
            ),
            MathTopic(
                "areas_related_to_circles",
                "Areas and circumferences of circles.",
                [
                    "A=πr²",
                    "C=2πr",
                ],
                "medium",
            ),
            MathTopic(
                "surface_areas_and_volumes",
                "Surface areas and volumes of solids.",
                [
                    "Sphere V=4/3πr³",
                    "Cylinder V=πr²h",
                    "Cone V=1/3πr²h",
                ],
                "hard",
            ),
            MathTopic(
                "statistics",
                "Mean, median, mode and grouped data.",
                [
                    "Mean = Σx/n",
                ],
                "medium",
            ),
            MathTopic(
                "probability",
                "Basic theoretical probability.",
                [
                    "P(E)=favourable outcomes/total outcomes",
                ],
                "easy",
            ),
        ]

        for topic in topics:

            self.topics[
                self._normalize_topic(topic.name)
            ] = topic

    # ========================================================
    # TOPIC ACCESS
    # ========================================================

    def get_topics(
        self,
    ) -> list[dict[str, Any]]:

        return [
            topic.to_dict()
            for topic in self.topics.values()
        ]

    def get_topic(
        self,
        topic: str,
    ) -> MathTopic | None:

        return self.topics.get(
            self._normalize_topic(topic)
        )

    def search_topics(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        query = query.casefold().strip()

        return [
            topic.to_dict()
            for topic in self.topics.values()
            if query in topic.name.casefold()
            or query in topic.description.casefold()
        ]

    # ========================================================
    # QUESTION GENERATION
    # ========================================================

    def generate_question(
        self,
        topic: str,
        *,
        difficulty: str = "medium",
    ) -> MathQuestion:

        normalized = self._normalize_topic(
            topic
        )

        difficulty = difficulty.casefold()

        generators = {
            "real_numbers":
                self._real_numbers_question,

            "polynomials":
                self._polynomial_question,

            "pair_of_linear_equations":
                self._linear_equation_question,

            "quadratic_equations":
                self._quadratic_question,

            "arithmetic_progressions":
                self._ap_question,

            "coordinate_geometry":
                self._coordinate_question,

            "trigonometry":
                self._trigonometry_question,

            "areas_related_to_circles":
                self._circle_area_question,

            "surface_areas_and_volumes":
                self._volume_question,

            "probability":
                self._probability_question,

            "statistics":
                self._statistics_question,
        }

        generator = generators.get(
            normalized
        )

        if generator is None:
            return self._generic_question(
                normalized,
                difficulty,
            )

        return generator(
            difficulty
        )

    def generate_questions(
        self,
        topic: str,
        count: int = 10,
        *,
        difficulty: str = "medium",
    ) -> list[MathQuestion]:

        if count <= 0:
            return []

        return [
            self.generate_question(
                topic,
                difficulty=difficulty,
            )
            for _ in range(count)
        ]

    # ========================================================
    # REAL NUMBERS
    # ========================================================

    def _real_numbers_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        a = self.random.randint(
            50,
            500,
        )

        b = self.random.randint(
            20,
            100,
        )

        answer = math.gcd(
            a,
            b,
        )

        return self._question(
            "real_numbers",
            f"Find the HCF of {a} and {b}.",
            answer,
            difficulty,
            f"The HCF of {a} and {b} is {answer}.",
            f"Use Euclid's division algorithm.",
        )

    # ========================================================
    # POLYNOMIALS
    # ========================================================

    def _polynomial_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        r1 = self.random.randint(
            -10,
            10,
        )

        r2 = self.random.randint(
            -10,
            10,
        )

        b = -(r1 + r2)
        c = r1 * r2

        expression = (
            f"x² + {b}x + {c}"
        )

        return self._question(
            "polynomials",
            (
                f"Find the zeros of "
                f"{expression}."
            ),
            sorted(
                [r1, r2]
            ),
            difficulty,
            (
                f"The polynomial factors as "
                f"(x - {r1})(x - {r2}), "
                f"so the zeros are {r1} and {r2}."
            ),
            "Factorise the quadratic.",
        )

    # ========================================================
    # LINEAR EQUATIONS
    # ========================================================

    def _linear_equation_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        x = self.random.randint(
            1,
            20,
        )

        a = self.random.randint(
            2,
            10,
        )

        b = self.random.randint(
            1,
            20,
        )

        c = a * x + b

        question = (
            f"Solve: {a}x + {b} = {c}"
        )

        return self._question(
            "pair_of_linear_equations",
            question,
            x,
            difficulty,
            (
                f"Subtract {b} from both sides: "
                f"{a}x = {c-b}. "
                f"Therefore x = {x}."
            ),
            f"First isolate {a}x.",
        )

    # ========================================================
    # QUADRATIC EQUATIONS
    # ========================================================

    def _quadratic_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        r1 = self.random.randint(
            1,
            10,
        )

        r2 = self.random.randint(
            1,
            10,
        )

        b = -(r1 + r2)
        c = r1 * r2

        question = (
            f"Solve: x² {self._signed(b)}x "
            f"+ {c} = 0"
        )

        return self._question(
            "quadratic_equations",
            question,
            sorted([r1, r2]),
            difficulty,
            (
                f"The equation factors into "
                f"(x-{r1})(x-{r2})=0. "
                f"Therefore x={r1} or x={r2}."
            ),
            "Try factorisation first.",
        )

    # ========================================================
    # ARITHMETIC PROGRESSIONS
    # ========================================================

    def _ap_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        a = self.random.randint(
            1,
            20,
        )

        d = self.random.randint(
            1,
            10,
        )

        n = self.random.randint(
            5,
            20,
        )

        answer = (
            a + (n - 1) * d
        )

        return self._question(
            "arithmetic_progressions",
            (
                f"Find the {n}th term of "
                f"the AP {a}, {a+d}, "
                f"{a+2*d}, ..."
            ),
            answer,
            difficulty,
            (
                f"Using aₙ = a + (n-1)d: "
                f"{a} + ({n}-1)×{d} = {answer}."
            ),
            "Use aₙ = a + (n-1)d.",
        )

    # ========================================================
    # COORDINATE GEOMETRY
    # ========================================================

    def _coordinate_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        x1 = self.random.randint(
            -10,
            10,
        )

        y1 = self.random.randint(
            -10,
            10,
        )

        x2 = self.random.randint(
            -10,
            10,
        )

        y2 = self.random.randint(
            -10,
            10,
        )

        dx = x2 - x1
        dy = y2 - y1

        distance_squared = (
            dx * dx + dy * dy
        )

        distance = math.sqrt(
            distance_squared
        )

        return self._question(
            "coordinate_geometry",
            (
                f"Find the distance between "
                f"({x1}, {y1}) and "
                f"({x2}, {y2})."
            ),
            self._simplify_sqrt(
                distance_squared
            ),
            difficulty,
            (
                "Use the distance formula: "
                f"√(({x2}-{x1})² + "
                f"({y2}-{y1})²) = "
                f"√{distance_squared} = "
                f"{self._simplify_sqrt(distance_squared)}."
            ),
            "Use the distance formula.",
        )

    # ========================================================
    # TRIGONOMETRY
    # ========================================================

    def _trigonometry_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        values = [
            ("30°", "sin", "1/2"),
            ("60°", "sin", "√3/2"),
            ("30°", "cos", "√3/2"),
            ("60°", "cos", "1/2"),
            ("45°", "sin", "1/√2"),
            ("45°", "cos", "1/√2"),
            ("45°", "tan", "1"),
            ("30°", "tan", "1/√3"),
            ("60°", "tan", "√3"),
        ]

        angle, function, answer = self.random.choice(
            values
        )

        return self._question(
            "trigonometry",
            f"Find {function} {angle}.",
            answer,
            difficulty,
            f"{function} {angle} = {answer}.",
            "Recall the standard trigonometric table.",
        )

    # ========================================================
    # CIRCLE
    # ========================================================

    def _circle_area_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        radius = self.random.randint(
            2,
            20,
        )

        area = (
            f"{radius}²π"
        )

        return self._question(
            "areas_related_to_circles",
            (
                f"Find the area of a circle "
                f"with radius {radius} cm. "
                f"Leave the answer in terms of π."
            ),
            area,
            difficulty,
            (
                f"A = πr² = π({radius})² "
                f"= {area} cm²."
            ),
            "Use A = πr².",
        )

    # ========================================================
    # VOLUMES
    # ========================================================

    def _volume_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        radius = self.random.randint(
            2,
            10,
        )

        height = self.random.randint(
            3,
            15,
        )

        volume = (
            f"{radius * radius * height}π"
        )

        return self._question(
            "surface_areas_and_volumes",
            (
                f"Find the volume of a cylinder "
                f"with radius {radius} cm and "
                f"height {height} cm. "
                f"Leave the answer in terms of π."
            ),
            volume,
            difficulty,
            (
                f"V = πr²h = π({radius})²"
                f"({height}) = {volume} cm³."
            ),
            "Use V = πr²h.",
        )

    # ========================================================
    # PROBABILITY
    # ========================================================

    def _probability_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        total = self.random.randint(
            4,
            20,
        )

        favourable = self.random.randint(
            1,
            total - 1,
        )

        fraction = self._reduce_fraction(
            favourable,
            total,
        )

        return self._question(
            "probability",
            (
                f"A bag contains {total} "
                f"equally likely outcomes. "
                f"{favourable} are favourable. "
                f"Find the probability of "
                f"the event."
            ),
            fraction,
            difficulty,
            (
                f"P(E) = favourable outcomes / "
                f"total outcomes = "
                f"{favourable}/{total} = "
                f"{fraction}."
            ),
            "Divide favourable outcomes by total outcomes.",
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def _statistics_question(
        self,
        difficulty: str,
    ) -> MathQuestion:

        numbers = [
            self.random.randint(
                1,
                50,
            )
            for _ in range(5)
        ]

        answer = sum(numbers) / len(
            numbers
        )

        return self._question(
            "statistics",
            (
                "Find the mean of: "
                + ", ".join(
                    map(str, numbers)
                )
            ),
            self._format_number(
                answer
            ),
            difficulty,
            (
                f"Mean = sum of observations / "
                f"number of observations = "
                f"{sum(numbers)}/{len(numbers)} "
                f"= {self._format_number(answer)}."
            ),
            "Add all observations and divide by their count.",
        )

    # ========================================================
    # GENERIC QUESTION
    # ========================================================

    def _generic_question(
        self,
        topic: str,
        difficulty: str,
    ) -> MathQuestion:

        return self._question(
            topic,
            (
                f"Explain the main concept of "
                f"{topic.replace('_', ' ')}."
            ),
            "",
            difficulty,
            (
                "This topic requires a conceptual "
                "explanation based on the current "
                "RENIX mathematics curriculum."
            ),
            (
                "Review the definitions, formulas "
                "and solved examples from this topic."
            ),
        )

    # ========================================================
    # ANSWER CHECKING
    # ========================================================

    def check_answer(
        self,
        question: MathQuestion,
        user_answer: Any,
    ) -> dict[str, Any]:

        expected = question.answer

        normalized_user = self._normalize_answer(
            user_answer
        )

        normalized_expected = self._normalize_answer(
            expected
        )

        correct = (
            normalized_user
            == normalized_expected
        )

        return {
            "correct": correct,
            "question_id": question.question_id,
            "topic": question.topic,
            "user_answer": user_answer,
            "correct_answer": expected,
            "accuracy": 100.0 if correct else 0.0,
            "explanation": question.explanation,
            "hint": None
            if correct
            else question.hint,
        }

    # ========================================================
    # HINTS
    # ========================================================

    def get_hint(
        self,
        question: MathQuestion,
    ) -> str:

        return question.hint

    def get_solution(
        self,
        question: MathQuestion,
    ) -> str:

        return question.explanation

    # ========================================================
    # FORMULAS
    # ========================================================

    def get_formulas(
        self,
        topic: str | None = None,
    ) -> list[str]:

        if topic is not None:

            found = self.get_topic(
                topic
            )

            if found is None:
                return []

            return list(
                found.formulas
            )

        formulas = []

        for topic_data in self.topics.values():

            formulas.extend(
                topic_data.formulas
            )

        return formulas

    # ========================================================
    # DIFFICULTY
    # ========================================================

    def classify_difficulty(
        self,
        question: str,
    ) -> str:

        text = question.casefold()

        hard_words = [
            "prove",
            "derive",
            "application",
            "word problem",
            "show that",
            "quadratic",
            "trigonometry",
            "height",
            "distance",
        ]

        easy_words = [
            "define",
            "what is",
            "find",
            "calculate",
            "state",
        ]

        if any(
            word in text
            for word in hard_words
        ):
            return "hard"

        if any(
            word in text
            for word in easy_words
        ):
            return "easy"

        return "medium"

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def recommend_topic(
        self,
        weak_topics: Iterable[str] | None = None,
    ) -> list[str]:

        if weak_topics:

            return list(
                weak_topics
            )

        topic_names = list(
            self.topics.keys()
        )

        self.random.shuffle(
            topic_names
        )

        return topic_names[:3]

    # ========================================================
    # QUESTION IDs
    # ========================================================

    def _question(
        self,
        topic: str,
        question: str,
        answer: Any,
        difficulty: str,
        explanation: str,
        hint: str,
        options: list[str] | None = None,
    ) -> MathQuestion:

        self.question_counter += 1

        return MathQuestion(
            question_id=(
                f"MATH-{self.question_counter:06d}"
            ),
            topic=topic,
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint=hint,
            options=options or [],
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _normalize_topic(
        topic: str,
    ) -> str:

        topic = topic.strip().casefold()

        aliases = {
            "real numbers": "real_numbers",
            "polynomials": "polynomials",
            "linear equations":
                "pair_of_linear_equations",
            "pair of linear equations":
                "pair_of_linear_equations",
            "quadratic":
                "quadratic_equations",
            "quadratic equations":
                "quadratic_equations",
            "ap":
                "arithmetic_progressions",
            "arithmetic progression":
                "arithmetic_progressions",
            "coordinate geometry":
                "coordinate_geometry",
            "trigonometry":
                "trigonometry",
            "circles":
                "circles",
            "circle":
                "circles",
            "probability":
                "probability",
            "statistics":
                "statistics",
            "surface area and volume":
                "surface_areas_and_volumes",
            "surface areas and volumes":
                "surface_areas_and_volumes",
        }

        return aliases.get(
            topic,
            topic.replace(" ", "_"),
        )

    @staticmethod
    def _signed(
        value: int,
    ) -> str:

        if value >= 0:
            return f"+ {value}"

        return f"- {abs(value)}"

    @staticmethod
    def _reduce_fraction(
        numerator: int,
        denominator: int,
    ) -> str:

        divisor = math.gcd(
            numerator,
            denominator,
        )

        numerator //= divisor
        denominator //= divisor

        if denominator == 1:
            return str(numerator)

        return f"{numerator}/{denominator}"

    @staticmethod
    def _simplify_sqrt(
        value: int,
    ) -> str:

        if value == 0:
            return "0"

        outside = 1
        inside = value

        factor = 2

        while factor * factor <= inside:

            square = factor * factor

            while inside % square == 0:

                inside //= square
                outside *= factor

            factor += 1

        if inside == 1:
            return str(outside)

        if outside == 1:
            return f"√{inside}"

        return f"{outside}√{inside}"

    @staticmethod
    def _format_number(
        value: float,
    ) -> str:

        if value.is_integer():
            return str(
                int(value)
            )

        return f"{value:.2f}".rstrip(
            "0"
        ).rstrip(".")

    @staticmethod
    def _normalize_answer(
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            (int, float),
        ):
            return round(
                float(value),
                6,
            )

        text = str(
            value
        ).strip().casefold()

        text = text.replace(
            " ",
            "",
        )

        text = text.replace(
            "−",
            "-",
        )

        # Normalize common equivalent
        # square-root notation.
        text = text.replace(
            "sqrt",
            "√",
        )

        # Remove unnecessary trailing .0
        text = re.sub(
            r"(\d+)\.0\b",
            r"\1",
            text,
        )

        return text


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def create_mathematics_engine(
    *,
    seed: int | None = None,
) -> MathematicsEngine:

    return MathematicsEngine(
        seed=seed
    )


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "MathQuestion",
    "MathTopic",
    "MathematicsEngine",
    "create_mathematics_engine",
]


