"""
RENIX Education — English Subject Engine

Provides:
- English chapter/topic management
- Grammar practice
- Vocabulary practice
- Reading comprehension
- Writing practice
- Question generation
- Answer checking
- Hints and explanations
- Difficulty classification
- Topic recommendations
- Literature and language skills support

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
class EnglishTopic:
    topic_id: str
    name: str
    category: str
    description: str
    difficulty: str = "medium"
    skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "difficulty": self.difficulty,
            "skills": list(self.skills),
        }


@dataclass
class EnglishQuestion:
    question_id: str
    topic: str
    category: str
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
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "explanation": self.explanation,
            "hint": self.hint,
            "options": list(self.options),
        }


# ============================================================
# ENGLISH ENGINE
# ============================================================


class EnglishEngine:
    """
    Main RENIX English subject engine.

    Supports grammar, vocabulary, reading, writing,
    literature and language skills.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:

        self.random = random.Random(seed)

        self.topics: dict[
            str,
            EnglishTopic,
        ] = {}

        self.question_counter = 0

        self._register_default_topics()

    # ========================================================
    # TOPICS
    # ========================================================

    def _register_default_topics(self) -> None:

        topics = [

            EnglishTopic(
                topic_id="tenses",
                name="Tenses",
                category="grammar",
                description=(
                    "Present, past and future tenses "
                    "and their forms."
                ),
                difficulty="medium",
                skills=[
                    "grammar",
                    "sentence formation",
                    "error correction",
                ],
            ),

            EnglishTopic(
                topic_id="modals",
                name="Modals",
                category="grammar",
                description=(
                    "Use of can, could, may, might, "
                    "must, should, would and related forms."
                ),
                difficulty="medium",
                skills=[
                    "grammar",
                    "sentence completion",
                ],
            ),

            EnglishTopic(
                topic_id="subject_verb_agreement",
                name="Subject-Verb Agreement",
                category="grammar",
                description=(
                    "Correct agreement between subjects "
                    "and verbs."
                ),
                difficulty="medium",
                skills=[
                    "grammar",
                    "error correction",
                ],
            ),

            EnglishTopic(
                topic_id="reported_speech",
                name="Reported Speech",
                category="grammar",
                description=(
                    "Conversion between direct and "
                    "indirect speech."
                ),
                difficulty="hard",
                skills=[
                    "grammar",
                    "transformation",
                ],
            ),

            EnglishTopic(
                topic_id="active_passive_voice",
                name="Active and Passive Voice",
                category="grammar",
                description=(
                    "Changing sentences between active "
                    "and passive voice."
                ),
                difficulty="hard",
                skills=[
                    "grammar",
                    "sentence transformation",
                ],
            ),

            EnglishTopic(
                topic_id="articles",
                name="Articles",
                category="grammar",
                description=(
                    "Correct use of a, an and the."
                ),
                difficulty="easy",
                skills=[
                    "grammar",
                    "sentence completion",
                ],
            ),

            EnglishTopic(
                topic_id="prepositions",
                name="Prepositions",
                category="grammar",
                description=(
                    "Correct use of words such as in, "
                    "on, at, by, with and from."
                ),
                difficulty="easy",
                skills=[
                    "grammar",
                    "usage",
                ],
            ),

            EnglishTopic(
                topic_id="conjunctions",
                name="Conjunctions",
                category="grammar",
                description=(
                    "Words used to connect words, "
                    "phrases and clauses."
                ),
                difficulty="easy",
                skills=[
                    "grammar",
                    "sentence formation",
                ],
            ),

            EnglishTopic(
                topic_id="vocabulary",
                name="Vocabulary",
                category="language",
                description=(
                    "Synonyms, antonyms, meanings, "
                    "word usage and contextual vocabulary."
                ),
                difficulty="medium",
                skills=[
                    "vocabulary",
                    "reading",
                ],
            ),

            EnglishTopic(
                topic_id="idioms",
                name="Idioms and Phrases",
                category="language",
                description=(
                    "Common English idioms and their meanings."
                ),
                difficulty="medium",
                skills=[
                    "vocabulary",
                    "language usage",
                ],
            ),

            EnglishTopic(
                topic_id="reading_comprehension",
                name="Reading Comprehension",
                category="reading",
                description=(
                    "Understanding passages, extracting "
                    "information and making inferences."
                ),
                difficulty="hard",
                skills=[
                    "reading",
                    "comprehension",
                    "inference",
                ],
            ),

            EnglishTopic(
                topic_id="writing",
                name="Writing Skills",
                category="writing",
                description=(
                    "Formal letters, reports, articles, "
                    "essays, speeches and creative writing."
                ),
                difficulty="hard",
                skills=[
                    "writing",
                    "organization",
                    "grammar",
                ],
            ),

            EnglishTopic(
                topic_id="literature",
                name="Literature",
                category="literature",
                description=(
                    "Poetry, prose, characters, themes, "
                    "literary devices and interpretation."
                ),
                difficulty="hard",
                skills=[
                    "literature",
                    "analysis",
                    "interpretation",
                ],
            ),

            EnglishTopic(
                topic_id="figures_of_speech",
                name="Figures of Speech",
                category="literature",
                description=(
                    "Simile, metaphor, personification, "
                    "alliteration, hyperbole and other devices."
                ),
                difficulty="medium",
                skills=[
                    "literature",
                    "analysis",
                ],
            ),
        ]

        for topic in topics:
            self.register_topic(topic)

    def register_topic(
        self,
        topic: EnglishTopic,
    ) -> None:

        self.topics[
            self._normalize_topic(
                topic.name
            )
        ] = topic

    # ========================================================
    # TOPIC ACCESS
    # ========================================================

    def get_topics(
        self,
        category: str | None = None,
    ) -> list[dict[str, Any]]:

        topics = list(
            self.topics.values()
        )

        if category:

            category = category.casefold()

            topics = [
                topic
                for topic in topics
                if topic.category.casefold()
                == category
            ]

        return [
            topic.to_dict()
            for topic in topics
        ]

    def get_topic(
        self,
        topic: str,
    ) -> EnglishTopic | None:

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
                f"{topic.category} "
                f"{topic.description}"
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
    ) -> EnglishQuestion:

        normalized = self._normalize_topic(
            topic
        )

        generators = {

            "tenses":
                self._tenses_question,

            "modals":
                self._modals_question,

            "subject_verb_agreement":
                self._subject_verb_question,

            "reported_speech":
                self._reported_speech_question,

            "active_and_passive_voice":
                self._voice_question,

            "articles":
                self._articles_question,

            "prepositions":
                self._preposition_question,

            "conjunctions":
                self._conjunction_question,

            "vocabulary":
                self._vocabulary_question,

            "idioms":
                self._idiom_question,

            "figures_of_speech":
                self._figure_of_speech_question,

            "reading_comprehension":
                self._reading_question,

            "writing":
                self._writing_question,

            "literature":
                self._literature_question,
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
    ) -> list[EnglishQuestion]:

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
    # TENSES
    # ========================================================

    def _tenses_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "She _____ to school every day.",
                "goes",
                "The subject 'she' takes the singular present form 'goes'.",
                ["go", "goes", "going", "gone"],
            ),
            (
                "They _____ football yesterday.",
                "played",
                "The word 'yesterday' indicates the simple past tense.",
                ["play", "plays", "played", "playing"],
            ),
            (
                "I _____ my homework before dinner.",
                "will finish",
                "The sentence refers to a future action.",
                ["finish", "finished", "will finish", "finishing"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(
                questions
            )
        )

        return self._question(
            topic="tenses",
            category="grammar",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="Look for the time indicator and subject.",
            options=options,
        )

    # ========================================================
    # MODALS
    # ========================================================

    def _modals_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "You _____ obey the rules.",
                "must",
                "Must expresses strong obligation.",
                ["can", "might", "must", "could"],
            ),
            (
                "_____ I come in, please?",
                "May",
                "May is commonly used to ask for permission.",
                ["Must", "May", "Should", "Would"],
            ),
            (
                "You _____ respect your elders.",
                "should",
                "Should expresses advice or moral duty.",
                ["should", "might", "can", "would"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(
                questions
            )
        )

        return self._question(
            topic="modals",
            category="grammar",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="Identify whether the sentence expresses permission, advice or obligation.",
            options=options,
        )

    # ========================================================
    # SUBJECT VERB AGREEMENT
    # ========================================================

    def _subject_verb_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "Neither of the boys _____ ready.",
                "is",
                "'Neither' is treated as singular in this construction.",
                ["are", "is", "were", "have"],
            ),
            (
                "The players _____ practising.",
                "are",
                "'Players' is plural, so the plural verb 'are' is required.",
                ["is", "are", "was", "has"],
            ),
            (
                "My brother _____ cricket every evening.",
                "plays",
                "The singular subject 'brother' takes 'plays'.",
                ["play", "plays", "playing", "played"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(
                questions
            )
        )

        return self._question(
            topic="subject_verb_agreement",
            category="grammar",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="Find the true subject before choosing the verb.",
            options=options,
        )

    # ========================================================
    # REPORTED SPEECH
    # ========================================================

    def _reported_speech_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                'Riya said, "I am tired."',
                "Riya said that she was tired.",
            ),
            (
                'He said, "I will help you."',
                "He said that he would help me.",
            ),
            (
                'Aman said, "I have finished my work."',
                "Aman said that he had finished his work.",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="reported_speech",
            category="grammar",
            question=(
                f"Change into reported speech: "
                f"{question}"
            ),
            answer=answer,
            difficulty=difficulty,
            explanation=(
                "Change the pronoun, tense and time reference "
                "where required while preserving the meaning."
            ),
            hint="Check the reporting verb, pronouns and tense.",
        )

    # ========================================================
    # ACTIVE / PASSIVE
    # ========================================================

    def _voice_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "The boy kicked the ball.",
                "The ball was kicked by the boy.",
            ),
            (
                "She writes a letter.",
                "A letter is written by her.",
            ),
            (
                "The teacher praised the student.",
                "The student was praised by the teacher.",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="active_and_passive_voice",
            category="grammar",
            question=(
                f"Change into passive voice: "
                f"{question}"
            ),
            answer=answer,
            difficulty=difficulty,
            explanation=(
                "Move the object into the subject position "
                "and use the appropriate form of 'be' "
                "with the past participle."
            ),
            hint="Find the object and make it the new subject.",
        )

    # ========================================================
    # ARTICLES
    # ========================================================

    def _articles_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "He is _____ honest man.",
                "an",
                "'Honest' begins with a vowel sound.",
                ["a", "an", "the", "no article"],
            ),
            (
                "I saw _____ elephant at the zoo.",
                "an",
                "'Elephant' begins with a vowel sound.",
                ["a", "an", "the", "no article"],
            ),
            (
                "_____ Sun rises in the east.",
                "The",
                "We use 'the' with unique objects such as the Sun.",
                ["A", "An", "The", "No article"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(
                questions
            )
        )

        return self._question(
            topic="articles",
            category="grammar",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="Focus on the sound and whether the noun is specific or unique.",
            options=options,
        )

    # ========================================================
    # PREPOSITIONS
    # ========================================================

    def _preposition_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "The book is _____ the table.",
                "on",
                "'On' is used when something is positioned on a surface.",
                ["in", "on", "at", "from"],
            ),
            (
                "She arrived _____ 8 o'clock.",
                "at",
                "'At' is used with specific clock times.",
                ["in", "on", "at", "by"],
            ),
            (
                "We live _____ Mumbai.",
                "in",
                "'In' is used with cities and larger areas.",
                ["at", "on", "in", "by"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(
                questions
            )
        )

        return self._question(
            topic="prepositions",
            category="grammar",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="Consider whether the sentence refers to time, place or position.",
            options=options,
        )

    # ========================================================
    # CONJUNCTIONS
    # ========================================================

    def _conjunction_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "I wanted to play, _____ it was raining.",
                "but",
                "'But' expresses contrast.",
                ["and", "but", "or", "so"],
            ),
            (
                "Study hard _____ you will succeed.",
                "and",
                "'And' connects two related ideas.",
                ["but", "and", "or", "because"],
            ),
            (
                "Hurry up, _____ you will miss the bus.",
                "or",
                "'Or' expresses an alternative or consequence.",
                ["and", "but", "or", "because"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(
                questions
            )
        )

        return self._question(
            topic="conjunctions",
            category="grammar",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="Identify the relationship between the two clauses.",
            options=options,
        )

    # ========================================================
    # VOCABULARY
    # ========================================================

    def _vocabulary_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "Choose the synonym of 'rapid'.",
                "fast",
                "'Rapid' means happening quickly or at high speed.",
                ["slow", "fast", "weak", "late"],
            ),
            (
                "Choose the antonym of 'ancient'.",
                "modern",
                "'Modern' is the opposite of 'ancient'.",
                ["old", "historic", "modern", "traditional"],
            ),
            (
                "Choose the synonym of 'brave'.",
                "courageous",
                "'Courageous' means brave.",
                ["cowardly", "courageous", "careless", "quiet"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(
                questions
            )
        )

        return self._question(
            topic="vocabulary",
            category="language",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="Think about the meaning of the target word.",
            options=options,
        )

    # ========================================================
    # IDIOMS
    # ========================================================

    def _idiom_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "What does the idiom 'once in a blue moon' mean?",
                "Very rarely",
            ),
            (
                "What does 'break the ice' mean?",
                "To start a friendly conversation",
            ),
            (
                "What does 'hit the nail on the head' mean?",
                "To say exactly the right thing",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="idioms",
            category="language",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=(
                f"The idiom means: {answer}."
            ),
            hint="Think about the figurative meaning, not the literal words.",
        )

    # ========================================================
    # FIGURES OF SPEECH
    # ========================================================

    def _figure_of_speech_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "Identify the figure of speech: "
                "'Her smile was like sunshine.'",
                "Simile",
                "The comparison uses the word 'like'.",
            ),
            (
                "Identify the figure of speech: "
                "'The wind whispered through the trees.'",
                "Personification",
                "The wind is given a human ability: whispering.",
            ),
            (
                "Identify the figure of speech: "
                "'He is a lion on the battlefield.'",
                "Metaphor",
                "The person is directly compared to a lion.",
            ),
            (
                "Identify the figure of speech: "
                "'Peter Piper picked a peck of pickled peppers.'",
                "Alliteration",
                "Repeated initial consonant sounds create alliteration.",
            ),
        ]

        question, answer, explanation = self.random.choice(
            questions
        )

        return self._question(
            topic="figures_of_speech",
            category="literature",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="Look for comparison, repetition or human qualities.",
        )

    # ========================================================
    # READING COMPREHENSION
    # ========================================================

    def _reading_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        passages = [

            (
                (
                    "Arjun had always believed that talent "
                    "was enough to succeed. After failing in "
                    "his first competition, he realised that "
                    "practice mattered just as much as talent. "
                    "He began practising every morning and "
                    "slowly improved."
                ),
                "What did Arjun realise after his failure?",
                "Practice mattered as much as talent.",
            ),

            (
                (
                    "The village once depended entirely on "
                    "a nearby river. Over time, pollution "
                    "made the water unsafe. The villagers "
                    "then worked together to clean the river "
                    "and reduce waste."
                ),
                "Why did the villagers work together?",
                "To clean the river and reduce pollution.",
            ),
        ]

        passage, question, answer = self.random.choice(
            passages
        )

        return self._question(
            topic="reading_comprehension",
            category="reading",
            question=(
                f"Read the passage:\n\n"
                f"{passage}\n\n"
                f"{question}"
            ),
            answer=answer,
            difficulty=difficulty,
            explanation=(
                "The answer should be supported directly "
                "by information in the passage."
            ),
            hint="Find the sentence in the passage that directly answers the question.",
        )

    # ========================================================
    # WRITING
    # ========================================================

    def _writing_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        prompts = [
            (
                "Write a formal letter to your principal "
                "requesting permission to organise a school event."
            ),
            (
                "Write an article on the importance of "
                "balancing academics and activities."
            ),
            (
                "Write a speech on the topic "
                "'The Power of Young Minds'."
            ),
            (
                "Write a report about a school sports event."
            ),
        ]

        prompt = self.random.choice(
            prompts
        )

        return self._question(
            topic="writing",
            category="writing",
            question=prompt,
            answer="writing_task",
            difficulty=difficulty,
            explanation=(
                "A strong answer should have appropriate "
                "format, clear organisation, relevant content, "
                "correct grammar and suitable vocabulary."
            ),
            hint=(
                "Plan the introduction, main points and conclusion "
                "before writing."
            ),
        )

    # ========================================================
    # LITERATURE
    # ========================================================

    def _literature_question(
        self,
        difficulty: str,
    ) -> EnglishQuestion:

        questions = [
            (
                "What is a theme in literature?",
                "A central idea or message explored in a literary work.",
            ),
            (
                "What is characterisation?",
                "The way an author develops and presents a character.",
            ),
            (
                "What is the setting of a story?",
                "The time and place in which a story occurs.",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="literature",
            category="literature",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=answer,
            hint="Focus on the basic elements of literary analysis.",
        )

    # ========================================================
    # GENERIC QUESTION
    # ========================================================

    def _generic_question(
        self,
        topic: str,
        difficulty: str,
    ) -> EnglishQuestion:

        data = self.get_topic(topic)

        if data is None:

            return self._question(
                topic=topic,
                category="english",
                question=(
                    f"Explain the main concept "
                    f"of {topic.replace('_', ' ')}."
                ),
                answer="",
                difficulty=difficulty,
                explanation=(
                    "Review the definitions, examples "
                    "and important rules related to this topic."
                ),
                hint=(
                    "Start with the definition and "
                    "then provide examples."
                ),
            )

        return self._question(
            topic=topic,
            category=data.category,
            question=(
                f"Explain {data.name} "
                f"in your own words."
            ),
            answer="conceptual",
            difficulty=difficulty,
            explanation=data.description,
            hint=(
                "Define the concept and explain "
                "its most important features."
            ),
        )

    # ========================================================
    # ANSWER CHECKING
    # ========================================================

    def check_answer(
        self,
        question: EnglishQuestion,
        user_answer: Any,
    ) -> dict[str, Any]:

        expected = self._normalize_answer(
            question.answer
        )

        actual = self._normalize_answer(
            user_answer
        )

        if expected == "writing_task":
            return self._evaluate_writing(
                question,
                str(user_answer),
            )

        if expected == "conceptual":
            return self._evaluate_conceptual(
                question,
                str(user_answer),
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
            "category": question.category,
            "user_answer": user_answer,
            "correct_answer": question.answer,
            "accuracy": (
                100.0
                if correct
                else 0.0
            ),
            "explanation": question.explanation,
            "hint": (
                None
                if correct
                else question.hint
            ),
        }

    # ========================================================
    # WRITING EVALUATION
    # ========================================================

    def _evaluate_writing(
        self,
        question: EnglishQuestion,
        answer: str,
    ) -> dict[str, Any]:

        words = self._word_count(
            answer
        )

        sentences = self._sentence_count(
            answer
        )

        grammar_score = self._grammar_score(
            answer
        )

        structure_score = (
            20.0
            if sentences >= 3
            else 10.0
            if sentences >= 1
            else 0.0
        )

        vocabulary_score = (
            min(
                20.0,
                len(set(
                    re.findall(
                        r"[A-Za-z]+",
                        answer.casefold(),
                    )
                )) / 5.0,
            )
        )

        length_score = (
            20.0
            if words >= 100
            else 15.0
            if words >= 60
            else 10.0
            if words >= 30
            else 5.0
            if words > 0
            else 0.0
        )

        score = min(
            100.0,
            grammar_score
            + structure_score
            + vocabulary_score
            + length_score,
        )

        return {
            "correct": score >= 60,
            "question_id": question.question_id,
            "topic": question.topic,
            "category": question.category,
            "user_answer": answer,
            "correct_answer": (
                "Open-ended writing task"
            ),
            "accuracy": round(
                score,
                2,
            ),
            "word_count": words,
            "sentence_count": sentences,
            "feedback": self._writing_feedback(
                words,
                sentences,
                score,
            ),
            "explanation": question.explanation,
            "hint": question.hint,
        }

    def _writing_feedback(
        self,
        words: int,
        sentences: int,
        score: float,
    ) -> list[str]:

        feedback = []

        if words < 60:
            feedback.append(
                "Develop your ideas with more supporting details."
            )

        if sentences < 3:
            feedback.append(
                "Use more complete sentences."
            )

        if score >= 80:
            feedback.append(
                "Strong overall response."
            )
        elif score >= 60:
            feedback.append(
                "Good attempt. Improve structure and vocabulary."
            )
        else:
            feedback.append(
                "Work on structure, grammar and development of ideas."
            )

        return feedback

    # ========================================================
    # CONCEPTUAL EVALUATION
    # ========================================================

    def _evaluate_conceptual(
        self,
        question: EnglishQuestion,
        answer: str,
    ) -> dict[str, Any]:

        if not answer.strip():

            score = 0.0

        else:

            words = set(
                re.findall(
                    r"[a-z]+",
                    answer.casefold(),
                )
            )

            score = min(
                100.0,
                40.0
                + min(
                    60.0,
                    len(words) * 3.0,
                ),
            )

        return {
            "correct": score >= 60,
            "question_id": question.question_id,
            "topic": question.topic,
            "category": question.category,
            "user_answer": answer,
            "accuracy": round(
                score,
                2,
            ),
            "feedback": (
                "Expand your explanation with "
                "definitions, examples and key points."
            ),
            "explanation": question.explanation,
            "hint": question.hint,
        }

    # ========================================================
    # HINTS / SOLUTIONS
    # ========================================================

    def get_hint(
        self,
        question: EnglishQuestion,
    ) -> str:

        return question.hint

    def get_solution(
        self,
        question: EnglishQuestion,
    ) -> str:

        return question.explanation

    # ========================================================
    # DIFFICULTY
    # ========================================================

    def classify_difficulty(
        self,
        question: str,
    ) -> str:

        text = question.casefold()

        hard_patterns = [
            "analyse",
            "analyze",
            "justify",
            "evaluate",
            "compare",
            "interpret",
            "explain why",
            "transform",
            "reported speech",
            "passive voice",
            "comprehension",
            "essay",
            "article",
            "speech",
        ]

        easy_patterns = [
            "define",
            "what is",
            "name",
            "identify",
            "choose",
            "meaning",
            "synonym",
            "antonym",
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
    # TOPIC RECOMMENDATIONS
    # ========================================================

    def recommend_topics(
        self,
        *,
        category: str | None = None,
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

        if category:

            category = category.casefold()

            topics = [
                topic
                for topic in topics
                if topic.category.casefold()
                == category
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
        category: str | None = None,
    ) -> EnglishTopic | None:

        topics = list(
            self.topics.values()
        )

        if category:

            category = category.casefold()

            topics = [
                topic
                for topic in topics
                if topic.category.casefold()
                == category
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
        category: str,
        question: str,
        answer: Any,
        difficulty: str,
        explanation: str,
        hint: str,
        options: list[str] | None = None,
    ) -> EnglishQuestion:

        self.question_counter += 1

        return EnglishQuestion(
            question_id=(
                f"ENG-{self.question_counter:06d}"
            ),
            topic=topic,
            category=category,
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

            "tense":
                "tenses",

            "tenses":
                "tenses",

            "modal":
                "modals",

            "modals":
                "modals",

            "subject verb agreement":
                "subject_verb_agreement",

            "subject-verb agreement":
                "subject_verb_agreement",

            "reported speech":
                "reported_speech",

            "active passive":
                "active_and_passive_voice",

            "active and passive voice":
                "active_and_passive_voice",

            "voice":
                "active_and_passive_voice",

            "articles":
                "articles",

            "article":
                "articles",

            "preposition":
                "prepositions",

            "prepositions":
                "prepositions",

            "conjunction":
                "conjunctions",

            "conjunctions":
                "conjunctions",

            "vocab":
                "vocabulary",

            "vocabulary":
                "vocabulary",

            "idioms":
                "idioms",

            "reading":
                "reading_comprehension",

            "reading comprehension":
                "reading_comprehension",

            "writing":
                "writing",

            "literature":
                "literature",

            "figures of speech":
                "figures_of_speech",

            "figure of speech":
                "figures_of_speech",
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

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = text.strip(
            " .!?\"'"
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

        common = (
            actual_words
            & expected_words
        )

        similarity = (
            len(common)
            / max(
                len(expected_words),
                1,
            )
        )

        return similarity >= 0.8

    # ========================================================
    # TEXT UTILITIES
    # ========================================================

    @staticmethod
    def _word_count(
        text: str,
    ) -> int:

        return len(
            re.findall(
                r"\b[\w'-]+\b",
                text,
            )
        )

    @staticmethod
    def _sentence_count(
        text: str,
    ) -> int:

        sentences = re.findall(
            r"[^.!?]+[.!?]+",
            text,
        )

        return len(
            sentences
        )

    @staticmethod
    def _grammar_score(
        text: str,
    ) -> float:

        if not text.strip():
            return 0.0

        score = 20.0

        if text[0].isupper():
            score += 10.0

        if text.rstrip().endswith(
            (".", "!", "?")
        ):
            score += 10.0

        repeated_spaces = re.search(
            r"\s{2,}",
            text,
        )

        if not repeated_spaces:
            score += 10.0

        return min(
            50.0,
            score,
        )


# ============================================================
# CONVENIENCE FACTORY
# ============================================================


def create_english_engine(
    *,
    seed: int | None = None,
) -> EnglishEngine:

    return EnglishEngine(
        seed=seed
    )


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "EnglishTopic",
    "EnglishQuestion",
    "EnglishEngine",
    "create_english_engine",
]


