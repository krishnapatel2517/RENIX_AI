"""
RENIX Education — Science Subject Engine

Provides:
- Science chapter management
- Physics, Chemistry and Biology topics
- Question generation
- Answer checking
- Hints and explanations
- Formula storage
- Definitions
- Difficulty classification
- Practice sessions
- Topic recommendations

Designed to integrate with:
    RENIX/education/study_manager.py
    RENIX/education/question_generator.py
    RENIX/education/answer_checker.py
    RENIX/education/progress_tracker.py
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class ScienceTopic:
    topic_id: str
    name: str
    subject_area: str
    description: str
    formulas: list[str] = field(default_factory=list)
    definitions: dict[str, str] = field(
        default_factory=dict
    )
    difficulty: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "subject_area": self.subject_area,
            "description": self.description,
            "formulas": list(self.formulas),
            "definitions": dict(self.definitions),
            "difficulty": self.difficulty,
        }


@dataclass
class ScienceQuestion:
    question_id: str
    topic: str
    subject_area: str
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
            "subject_area": self.subject_area,
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "explanation": self.explanation,
            "hint": self.hint,
            "options": list(self.options),
        }


# ============================================================
# SCIENCE ENGINE
# ============================================================


class ScienceEngine:
    """
    Main RENIX science engine.

    Example:

        science = ScienceEngine()

        question = science.generate_question(
            "electricity"
        )

        result = science.check_answer(
            question,
            "10"
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
            ScienceTopic,
        ] = {}

        self.question_counter = 0

        self._register_default_topics()

    # ========================================================
    # TOPIC REGISTRATION
    # ========================================================

    def _register_default_topics(
        self,
    ) -> None:

        topics = [

            # ------------------------------------------------
            # PHYSICS
            # ------------------------------------------------

            ScienceTopic(
                topic_id="light_reflection",
                name="Light – Reflection and Refraction",
                subject_area="physics",
                description=(
                    "Reflection, refraction, mirrors, lenses "
                    "and ray diagrams."
                ),
                formulas=[
                    "1/f = 1/v + 1/u",
                    "m = hᵢ/hₒ = -v/u",
                    "n = sin i / sin r",
                ],
                definitions={
                    "reflection":
                        "The bouncing back of light from a surface.",
                    "refraction":
                        "The bending of light when it passes from one medium to another.",
                    "focal_length":
                        "The distance between the optical centre or pole and the principal focus.",
                },
                difficulty="medium",
            ),

            ScienceTopic(
                topic_id="human_eye",
                name="Human Eye and Colourful World",
                subject_area="physics",
                description=(
                    "Structure of the human eye, accommodation, "
                    "defects of vision and dispersion."
                ),
                formulas=[
                    "P = 1/f",
                ],
                definitions={
                    "accommodation":
                        "The ability of the eye lens to adjust its focal length.",
                    "myopia":
                        "A defect in which nearby objects are clear but distant objects appear blurred.",
                    "hypermetropia":
                        "A defect in which distant objects are clear but nearby objects appear blurred.",
                },
                difficulty="medium",
            ),

            ScienceTopic(
                topic_id="electricity",
                name="Electricity",
                subject_area="physics",
                description=(
                    "Electric current, potential difference, resistance, "
                    "Ohm's law and electrical power."
                ),
                formulas=[
                    "V = IR",
                    "R = V/I",
                    "P = VI",
                    "P = I²R",
                    "P = V²/R",
                    "E = Pt",
                ],
                definitions={
                    "current":
                        "The rate of flow of electric charge.",
                    "voltage":
                        "The potential difference between two points.",
                    "resistance":
                        "The opposition offered to the flow of electric current.",
                },
                difficulty="medium",
            ),

            ScienceTopic(
                topic_id="magnetic_effects",
                name="Magnetic Effects of Electric Current",
                subject_area="physics",
                description=(
                    "Magnetic fields, electromagnets, electric motors "
                    "and electromagnetic induction."
                ),
                formulas=[],
                definitions={
                    "magnetic_field":
                        "The region around a magnet or current-carrying conductor where magnetic effects can be observed.",
                    "electromagnet":
                        "A temporary magnet produced by passing current through a coil.",
                    "electromagnetic_induction":
                        "The production of electric current due to a changing magnetic field.",
                },
                difficulty="medium",
            ),

            # ------------------------------------------------
            # CHEMISTRY
            # ------------------------------------------------

            ScienceTopic(
                topic_id="chemical_reactions",
                name="Chemical Reactions and Equations",
                subject_area="chemistry",
                description=(
                    "Types of chemical reactions, balancing equations "
                    "and oxidation-reduction."
                ),
                formulas=[],
                definitions={
                    "oxidation":
                        "Addition of oxygen, removal of hydrogen, or loss of electrons.",
                    "reduction":
                        "Removal of oxygen, addition of hydrogen, or gain of electrons.",
                    "displacement":
                        "A reaction in which a more reactive element displaces a less reactive element.",
                },
                difficulty="medium",
            ),

            ScienceTopic(
                topic_id="acids_bases_salts",
                name="Acids, Bases and Salts",
                subject_area="chemistry",
                description=(
                    "Properties of acids and bases, indicators, pH "
                    "and common salts."
                ),
                formulas=[
                    "pH = -log[H⁺]",
                ],
                definitions={
                    "acid":
                        "A substance that produces hydrogen ions in aqueous solution.",
                    "base":
                        "A substance that produces hydroxide ions in aqueous solution.",
                    "neutralisation":
                        "A reaction between an acid and a base producing salt and water.",
                },
                difficulty="medium",
            ),

            ScienceTopic(
                topic_id="metals_nonmetals",
                name="Metals and Non-metals",
                subject_area="chemistry",
                description=(
                    "Physical and chemical properties of metals and "
                    "non-metals, reactivity and corrosion."
                ),
                formulas=[],
                definitions={
                    "corrosion":
                        "The gradual destruction of a metal due to chemical reactions with its environment.",
                    "alloy":
                        "A homogeneous mixture of two or more metals or a metal and a non-metal.",
                    "reactivity_series":
                        "An arrangement of metals in decreasing order of reactivity.",
                },
                difficulty="medium",
            ),

            ScienceTopic(
                topic_id="carbon_compounds",
                name="Carbon and Its Compounds",
                subject_area="chemistry",
                description=(
                    "Carbon bonding, hydrocarbons, functional groups, "
                    "ethanol, ethanoic acid and soaps."
                ),
                formulas=[],
                definitions={
                    "covalent_bond":
                        "A chemical bond formed by sharing electrons between atoms.",
                    "homologous_series":
                        "A series of organic compounds having the same functional group and similar chemical properties.",
                    "functional_group":
                        "An atom or group of atoms responsible for characteristic chemical properties of an organic compound.",
                },
                difficulty="hard",
            ),

            # ------------------------------------------------
            # BIOLOGY
            # ------------------------------------------------

            ScienceTopic(
                topic_id="life_processes",
                name="Life Processes",
                subject_area="biology",
                description=(
                    "Nutrition, respiration, transportation and "
                    "excretion in living organisms."
                ),
                formulas=[],
                definitions={
                    "nutrition":
                        "The process by which organisms obtain and use nutrients.",
                    "respiration":
                        "The process of releasing energy from food.",
                    "excretion":
                        "The removal of metabolic waste from an organism.",
                },
                difficulty="medium",
            ),

            ScienceTopic(
                topic_id="control_coordination",
                name="Control and Coordination",
                subject_area="biology",
                description=(
                    "Nervous system, hormones, reflex actions and "
                    "coordination in plants."
                ),
                formulas=[],
                definitions={
                    "reflex_action":
                        "A rapid, automatic response to a stimulus.",
                    "hormone":
                        "A chemical messenger produced by endocrine glands.",
                    "neuron":
                        "The structural and functional unit of the nervous system.",
                },
                difficulty="medium",
            ),

            ScienceTopic(
                topic_id="reproduction",
                name="How Do Organisms Reproduce?",
                subject_area="biology",
                description=(
                    "Asexual and sexual reproduction in organisms "
                    "and reproductive systems."
                ),
                formulas=[],
                definitions={
                    "asexual_reproduction":
                        "Reproduction involving a single parent without fusion of gametes.",
                    "sexual_reproduction":
                        "Reproduction involving fusion of male and female gametes.",
                    "fertilisation":
                        "Fusion of male and female gametes to form a zygote.",
                },
                difficulty="medium",
            ),

            ScienceTopic(
                topic_id="heredity",
                name="Heredity",
                subject_area="biology",
                description=(
                    "Inheritance, Mendelian genetics, traits and "
                    "variation."
                ),
                formulas=[
                    "Phenotypic ratio in a typical monohybrid cross = 3:1",
                ],
                definitions={
                    "heredity":
                        "The transmission of traits from parents to offspring.",
                    "gene":
                        "A unit of heredity responsible for a particular trait.",
                    "variation":
                        "Differences in characteristics among individuals of the same species.",
                },
                difficulty="hard",
            ),

            ScienceTopic(
                topic_id="environment",
                name="Our Environment",
                subject_area="biology",
                description=(
                    "Ecosystems, food chains, food webs, trophic levels "
                    "and environmental management."
                ),
                formulas=[],
                definitions={
                    "ecosystem":
                        "A functional unit consisting of living organisms and their physical environment.",
                    "food_chain":
                        "A sequence showing the transfer of food and energy between organisms.",
                    "biodegradable":
                        "A substance that can be broken down naturally by microorganisms.",
                },
                difficulty="easy",
            ),

            ScienceTopic(
                topic_id="sustainable_management",
                name="Sustainable Management of Natural Resources",
                subject_area="biology",
                description=(
                    "Conservation and sustainable use of forests, wildlife, "
                    "water and other natural resources."
                ),
                formulas=[],
                definitions={
                    "sustainable_development":
                        "Development that meets present needs without compromising the ability of future generations to meet their needs.",
                    "conservation":
                        "Protection and careful management of natural resources.",
                },
                difficulty="medium",
            ),
        ]

        for topic in topics:
            self.register_topic(topic)

    def register_topic(
        self,
        topic: ScienceTopic,
    ) -> None:

        key = self._normalize_topic(
            topic.name
        )

        self.topics[key] = topic

    # ========================================================
    # TOPIC ACCESS
    # ========================================================

    def get_topics(
        self,
        subject_area: str | None = None,
    ) -> list[dict[str, Any]]:

        topics = list(
            self.topics.values()
        )

        if subject_area:
            area = subject_area.casefold()

            topics = [
                topic
                for topic in topics
                if topic.subject_area.casefold()
                == area
            ]

        return [
            topic.to_dict()
            for topic in topics
        ]

    def get_topic(
        self,
        topic: str,
    ) -> ScienceTopic | None:

        return self.topics.get(
            self._normalize_topic(topic)
        )

    def search_topics(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        query = query.casefold().strip()

        results = []

        for topic in self.topics.values():

            searchable = (
                f"{topic.name} "
                f"{topic.description} "
                f"{topic.subject_area}"
            ).casefold()

            if query in searchable:
                results.append(
                    topic.to_dict()
                )

        return results

    # ========================================================
    # QUESTION GENERATION
    # ========================================================

    def generate_question(
        self,
        topic: str,
        *,
        difficulty: str = "medium",
    ) -> ScienceQuestion:

        normalized = self._normalize_topic(
            topic
        )

        generators = {

            "electricity":
                self._electricity_question,

            "light_reflection":
                self._light_question,

            "human_eye":
                self._human_eye_question,

            "chemical_reactions":
                self._chemical_reaction_question,

            "acids_bases_salts":
                self._acid_base_question,

            "metals_nonmetals":
                self._metals_question,

            "carbon_compounds":
                self._carbon_question,

            "life_processes":
                self._life_process_question,

            "control_coordination":
                self._control_question,

            "reproduction":
                self._reproduction_question,

            "heredity":
                self._heredity_question,

            "environment":
                self._environment_question,

            "sustainable_management":
                self._sustainable_question,
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
    ) -> list[ScienceQuestion]:

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
    # PHYSICS — ELECTRICITY
    # ========================================================

    def _electricity_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        resistance = self.random.randint(
            2,
            20,
        )

        current = self.random.randint(
            1,
            10,
        )

        voltage = resistance * current

        return self._question(
            topic="electricity",
            subject_area="physics",
            question=(
                f"A current of {current} A flows "
                f"through a resistor of {resistance} Ω. "
                f"Find the potential difference."
            ),
            answer=f"{voltage} V",
            difficulty=difficulty,
            explanation=(
                f"Using Ohm's law V = IR, "
                f"V = {current} × {resistance} "
                f"= {voltage} V."
            ),
            hint="Use V = IR.",
        )

    # ========================================================
    # PHYSICS — LIGHT
    # ========================================================

    def _light_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        focal_length = self.random.choice(
            [5, 10, 15, 20, 25]
        )

        question = (
            f"A lens has a focal length of "
            f"{focal_length} cm. What is its "
            f"focal length in metres?"
        )

        answer = f"{focal_length / 100:g} m"

        return self._question(
            topic="light_reflection",
            subject_area="physics",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"1 m = 100 cm. Therefore "
                f"{focal_length} cm = "
                f"{focal_length / 100:g} m."
            ),
            hint="Convert centimetres into metres.",
        )

    # ========================================================
    # PHYSICS — HUMAN EYE
    # ========================================================

    def _human_eye_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "Which part of the eye controls "
                "the amount of light entering it?",
                "Iris",
                "The iris controls the size of the pupil.",
            ),
            (
                "Which part of the eye acts as a "
                "screen on which the image is formed?",
                "Retina",
                "The retina contains light-sensitive cells.",
            ),
            (
                "Which defect of vision is corrected "
                "using a concave lens?",
                "Myopia",
                "A concave lens helps a myopic eye focus distant objects correctly.",
            ),
        ]

        question, answer, explanation = self.random.choice(
            questions
        )

        return self._question(
            topic="human_eye",
            subject_area="physics",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="Recall the structure and defects of the human eye.",
        )

    # ========================================================
    # CHEMISTRY — REACTIONS
    # ========================================================

    def _chemical_reaction_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "What type of reaction occurs when "
                "one element replaces another element "
                "in a compound?",
                "Displacement reaction",
            ),
            (
                "What is the reaction between an acid "
                "and a base called?",
                "Neutralisation reaction",
            ),
            (
                "What type of reaction involves "
                "addition of oxygen?",
                "Oxidation",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="chemical_reactions",
            subject_area="chemistry",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The correct answer is {answer}."
            ),
            hint="Recall the definitions of common reaction types.",
        )

    # ========================================================
    # CHEMISTRY — ACIDS BASES
    # ========================================================

    def _acid_base_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "A solution has pH 2. Is it acidic or basic?",
                "Acidic",
            ),
            (
                "A solution has pH 12. Is it acidic or basic?",
                "Basic",
            ),
            (
                "What is formed when an acid reacts "
                "with a base?",
                "Salt and water",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="acids_bases_salts",
            subject_area="chemistry",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The correct answer is {answer}."
            ),
            hint="Remember that pH below 7 is acidic and pH above 7 is basic.",
        )

    # ========================================================
    # CHEMISTRY — METALS
    # ========================================================

    def _metals_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "Which metal is commonly stored under kerosene?",
                "Sodium",
            ),
            (
                "What is the gradual destruction of metals "
                "due to environmental reactions called?",
                "Corrosion",
            ),
            (
                "What is a mixture of metals called?",
                "Alloy",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="metals_nonmetals",
            subject_area="chemistry",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The correct answer is {answer}."
            ),
            hint="Recall the properties and uses of metals.",
        )

    # ========================================================
    # CHEMISTRY — CARBON
    # ========================================================

    def _carbon_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "What type of bond is generally formed "
                "between carbon atoms in organic compounds?",
                "Covalent bond",
            ),
            (
                "What functional group is present in alcohols?",
                "-OH",
            ),
            (
                "What is the molecular formula of methane?",
                "CH4",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="carbon_compounds",
            subject_area="chemistry",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The correct answer is {answer}."
            ),
            hint="Think about carbon's tetravalency and common functional groups.",
        )

    # ========================================================
    # BIOLOGY — LIFE PROCESSES
    # ========================================================

    def _life_process_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "Which organ pumps blood throughout "
                "the human body?",
                "Heart",
            ),
            (
                "Which organ is primarily responsible "
                "for filtering blood and producing urine?",
                "Kidney",
            ),
            (
                "Which green pigment is essential "
                "for photosynthesis?",
                "Chlorophyll",
            ),
            (
                "Which gas is released during photosynthesis?",
                "Oxygen",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="life_processes",
            subject_area="biology",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The correct answer is {answer}."
            ),
            hint="Recall the major human life processes and plant nutrition.",
        )

    # ========================================================
    # BIOLOGY — CONTROL
    # ========================================================

    def _control_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "What is the basic structural and "
                "functional unit of the nervous system?",
                "Neuron",
            ),
            (
                "What is a rapid automatic response "
                "to a stimulus called?",
                "Reflex action",
            ),
            (
                "Which hormone regulates blood sugar "
                "levels in the human body?",
                "Insulin",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="control_coordination",
            subject_area="biology",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The correct answer is {answer}."
            ),
            hint="Recall the nervous system and endocrine system.",
        )

    # ========================================================
    # BIOLOGY — REPRODUCTION
    # ========================================================

    def _reproduction_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "What is the fusion of male and female "
                "gametes called?",
                "Fertilisation",
            ),
            (
                "How many parents are involved in "
                "asexual reproduction?",
                "One",
            ),
            (
                "What is the first cell formed after "
                "fertilisation called?",
                "Zygote",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="reproduction",
            subject_area="biology",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The correct answer is {answer}."
            ),
            hint="Recall the basic stages of reproduction.",
        )

    # ========================================================
    # BIOLOGY — HEREDITY
    # ========================================================

    def _heredity_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "What is the basic unit of heredity?",
                "Gene",
            ),
            (
                "Who is known as the father of genetics?",
                "Gregor Mendel",
            ),
            (
                "What term describes differences between "
                "individuals of the same species?",
                "Variation",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="heredity",
            subject_area="biology",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The correct answer is {answer}."
            ),
            hint="Recall Mendelian genetics and inheritance.",
        )

    # ========================================================
    # BIOLOGY — ENVIRONMENT
    # ========================================================

    def _environment_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "What is the sequence of organisms "
                "through which food and energy pass called?",
                "Food chain",
            ),
            (
                "What type of substances can be broken "
                "down naturally by microorganisms?",
                "Biodegradable substances",
            ),
            (
                "What is the main source of energy "
                "for most ecosystems?",
                "Sun",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="environment",
            subject_area="biology",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The correct answer is {answer}."
            ),
            hint="Think about ecosystems and energy flow.",
        )

    # ========================================================
    # BIOLOGY — SUSTAINABILITY
    # ========================================================

    def _sustainable_question(
        self,
        difficulty: str,
    ) -> ScienceQuestion:

        questions = [
            (
                "What does sustainable development "
                "aim to protect for future generations?",
                "Natural resources",
            ),
            (
                "What does conservation mean?",
                "Protection and careful management of natural resources",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="sustainable_management",
            subject_area="biology",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The correct answer is {answer}."
            ),
            hint="Think about responsible use of natural resources.",
        )

    # ========================================================
    # GENERIC QUESTION
    # ========================================================

    def _generic_question(
        self,
        topic: str,
        difficulty: str,
    ) -> ScienceQuestion:

        data = self.get_topic(topic)

        if data is None:

            return self._question(
                topic=topic,
                subject_area="science",
                question=(
                    f"Explain the main concept "
                    f"of {topic.replace('_', ' ')}."
                ),
                answer="",
                difficulty=difficulty,
                explanation=(
                    "Review the relevant definitions, "
                    "concepts and examples."
                ),
                hint=(
                    "Start with the definition and "
                    "then explain the key concept."
                ),
            )

        return self._question(
            topic=topic,
            subject_area=data.subject_area,
            question=(
                f"Explain {data.name} "
                f"in your own words."
            ),
            answer="conceptual",
            difficulty=difficulty,
            explanation=data.description,
            hint=(
                "Start with the definition, "
                "then explain the main points."
            ),
        )

    # ========================================================
    # ANSWER CHECKING
    # ========================================================

    def check_answer(
        self,
        question: ScienceQuestion,
        user_answer: Any,
    ) -> dict[str, Any]:

        expected = self._normalize_answer(
            question.answer
        )

        actual = self._normalize_answer(
            user_answer
        )

        correct = (
            actual == expected
            or self._semantic_match(
                actual,
                expected,
            )
        )

        return {
            "correct": correct,
            "question_id": question.question_id,
            "topic": question.topic,
            "subject_area": question.subject_area,
            "user_answer": user_answer,
            "correct_answer": question.answer,
            "accuracy": 100.0 if correct else 0.0,
            "explanation": question.explanation,
            "hint": (
                None
                if correct
                else question.hint
            ),
        }

    # ========================================================
    # HINTS / SOLUTIONS
    # ========================================================

    def get_hint(
        self,
        question: ScienceQuestion,
    ) -> str:

        return question.hint

    def get_solution(
        self,
        question: ScienceQuestion,
    ) -> str:

        return question.explanation

    # ========================================================
    # FORMULAS
    # ========================================================

    def get_formulas(
        self,
        topic: str | None = None,
    ) -> list[str]:

        if topic:

            data = self.get_topic(
                topic
            )

            if data is None:
                return []

            return list(
                data.formulas
            )

        formulas = []

        for data in self.topics.values():

            formulas.extend(
                data.formulas
            )

        return formulas

    # ========================================================
    # DEFINITIONS
    # ========================================================

    def get_definitions(
        self,
        topic: str | None = None,
    ) -> dict[str, str]:

        if topic:

            data = self.get_topic(
                topic
            )

            if data is None:
                return {}

            return dict(
                data.definitions
            )

        definitions: dict[
            str,
            str,
        ] = {}

        for data in self.topics.values():

            definitions.update(
                data.definitions
            )

        return definitions

    # ========================================================
    # DIFFICULTY CLASSIFICATION
    # ========================================================

    def classify_difficulty(
        self,
        question: str,
    ) -> str:

        text = question.casefold()

        hard_patterns = [
            "derive",
            "prove",
            "explain why",
            "analyse",
            "compare",
            "calculate",
            "numerical",
            "application",
            "reason",
            "justify",
        ]

        easy_patterns = [
            "define",
            "what is",
            "name",
            "state",
            "identify",
            "which",
        ]

        if any(
            pattern in text
            for pattern in hard_patterns
        ):
            return "hard"

        if any(
            pattern in text
            for pattern in easy_patterns
        ):
            return "easy"

        return "medium"

    # ========================================================
    # TOPIC RECOMMENDATION
    # ========================================================

    def recommend_topics(
        self,
        *,
        subject_area: str | None = None,
        weak_topics: Iterable[str] | None = None,
        count: int = 3,
    ) -> list[str]:

        if weak_topics:

            return list(
                weak_topics
            )[:count]

        topics = list(
            self.topics.values()
        )

        if subject_area:

            area = subject_area.casefold()

            topics = [
                topic
                for topic in topics
                if topic.subject_area.casefold()
                == area
            ]

        self.random.shuffle(
            topics
        )

        return [
            topic.topic_id
            for topic in topics[:count]
        ]

    # ========================================================
    # RANDOM TOPIC
    # ========================================================

    def random_topic(
        self,
        subject_area: str | None = None,
    ) -> ScienceTopic | None:

        topics = list(
            self.topics.values()
        )

        if subject_area:

            area = subject_area.casefold()

            topics = [
                topic
                for topic in topics
                if topic.subject_area.casefold()
                == area
            ]

        if not topics:
            return None

        return self.random.choice(
            topics
        )

    # ========================================================
    # QUESTION FACTORY
    # ========================================================

    def _question(
        self,
        *,
        topic: str,
        subject_area: str,
        question: str,
        answer: Any,
        difficulty: str,
        explanation: str,
        hint: str,
        options: list[str] | None = None,
    ) -> ScienceQuestion:

        self.question_counter += 1

        return ScienceQuestion(
            question_id=(
                f"SCI-{self.question_counter:06d}"
            ),
            topic=topic,
            subject_area=subject_area,
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint=hint,
            options=options or [],
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_topic(
        topic: str,
    ) -> str:

        value = (
            topic
            .strip()
            .casefold()
        )

        aliases = {

            "electricity":
                "electricity",

            "light":
                "light_reflection",

            "reflection":
                "light_reflection",

            "refraction":
                "light_reflection",

            "human eye":
                "human_eye",

            "eye":
                "human_eye",

            "chemical reactions":
                "chemical_reactions",

            "reactions":
                "chemical_reactions",

            "acids bases salts":
                "acids_bases_salts",

            "acids":
                "acids_bases_salts",

            "bases":
                "acids_bases_salts",

            "metals":
                "metals_nonmetals",

            "non metals":
                "metals_nonmetals",

            "carbon":
                "carbon_compounds",

            "carbon compounds":
                "carbon_compounds",

            "life processes":
                "life_processes",

            "control and coordination":
                "control_coordination",

            "control coordination":
                "control_coordination",

            "reproduction":
                "reproduction",

            "heredity":
                "heredity",

            "environment":
                "environment",

            "sustainable management":
                "sustainable_management",
        }

        if value in aliases:
            return aliases[value]

        return value.replace(
            " ",
            "_",
        )

    # ========================================================
    # ANSWER NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_answer(
        answer: Any,
    ) -> str:

        if answer is None:
            return ""

        text = str(
            answer
        ).strip().casefold()

        text = text.replace(
            "−",
            "-",
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = text.strip(
            " ."
        )

        return text

    # ========================================================
    # SEMANTIC MATCHING
    # ========================================================

    @staticmethod
    def _semantic_match(
        actual: str,
        expected: str,
    ) -> bool:

        if not actual or not expected:
            return False

        if actual == expected:
            return True

        actual_words = set(
            re.findall(
                r"[a-z0-9]+",
                actual,
            )
        )

        expected_words = set(
            re.findall(
                r"[a-z0-9]+",
                expected,
            )
        )

        if not actual_words or not expected_words:
            return False

        intersection = (
            actual_words
            & expected_words
        )

        similarity = (
            len(intersection)
            / max(
                len(expected_words),
                1,
            )
        )

        return similarity >= 0.8


# ============================================================
# CONVENIENCE FACTORY
# ============================================================


def create_science_engine(
    *,
    seed: int | None = None,
) -> ScienceEngine:

    return ScienceEngine(
        seed=seed
    )


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "ScienceTopic",
    "ScienceQuestion",
    "ScienceEngine",
    "create_science_engine",
]


