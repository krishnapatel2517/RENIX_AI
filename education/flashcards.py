"""
RENIX Education — Flashcards Engine

Handles:
- Flashcard creation and management
- Decks
- Study sessions
- Spaced repetition
- Difficulty ratings
- Confidence tracking
- Due-card scheduling
- Review history
- Weak-card detection
- Progress statistics
- Import/export
- Storage and EventBus integration

Designed to integrate with:
    education/study_manager.py
    education/progress_tracker.py
    education/quiz_engine.py
    memory/memory_manager.py
    database/repositories.py
    core/event_bus.py
"""

from __future__ import annotations

import logging
import threading
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

RATINGS = {
    "again": 0,
    "hard": 1,
    "good": 2,
    "easy": 3,
}

CARD_STATUSES = {
    "new",
    "learning",
    "review",
    "suspended",
}

DECK_STATUSES = {
    "active",
    "archived",
}


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class Flashcard:
    """Represents one flashcard."""

    id: str

    front: str
    back: str

    deck_id: str

    subject: str = ""
    chapter: str = ""
    topic: str = ""

    hint: str = ""
    explanation: str = ""

    tags: list[str] = field(
        default_factory=list
    )

    status: str = "new"

    difficulty: float = 0.0

    repetitions: int = 0

    lapses: int = 0

    ease_factor: float = 2.5

    interval_days: int = 0

    due_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    last_reviewed_at: str | None = None

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlashcardDeck:
    """Represents a collection of flashcards."""

    id: str

    name: str

    description: str = ""

    subject: str = ""

    chapter: str = ""

    topic: str = ""

    status: str = "active"

    card_ids: list[str] = field(
        default_factory=list
    )

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    tags: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlashcardReview:
    """Represents a single card review."""

    id: str

    card_id: str

    rating: str

    previous_interval_days: int

    new_interval_days: int

    previous_ease_factor: float

    new_ease_factor: float

    response_time_seconds: float | None = None

    reviewed_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlashcardStatistics:
    """Aggregated flashcard statistics."""

    total_decks: int = 0

    total_cards: int = 0

    new_cards: int = 0

    learning_cards: int = 0

    review_cards: int = 0

    suspended_cards: int = 0

    due_cards: int = 0

    total_reviews: int = 0

    correct_reviews: int = 0

    average_interval_days: float = 0.0

    average_ease_factor: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# FLASHCARD ENGINE
# ============================================================


class FlashcardEngine:
    """
    Main RENIX flashcard system.

    Example:

        engine = FlashcardEngine()

        deck = engine.create_deck(
            name="Biology",
            subject="Science",
        )

        card = engine.create_card(
            deck_id=deck.id,
            front="What is photosynthesis?",
            back="The process by which plants..."
        )

        due_cards = engine.get_due_cards(
            deck.id
        )

        engine.review_card(
            card.id,
            "good"
        )
    """

    def __init__(
        self,
        *,
        storage: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:

        self.storage = storage
        self.event_bus = event_bus

        self._decks: dict[
            str,
            FlashcardDeck,
        ] = {}

        self._cards: dict[
            str,
            Flashcard,
        ] = {}

        self._reviews: dict[
            str,
            FlashcardReview,
        ] = {}

        self._lock = threading.RLock()

        logger.info(
            "RENIX FlashcardEngine initialized."
        )

    # ========================================================
    # DECK MANAGEMENT
    # ========================================================

    def create_deck(
        self,
        *,
        name: str,
        description: str = "",
        subject: str = "",
        chapter: str = "",
        topic: str = "",
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FlashcardDeck:

        name = self._required_text(
            name,
            "name",
        )

        deck = FlashcardDeck(
            id=self._new_deck_id(),
            name=name,
            description=description.strip(),
            subject=subject.strip(),
            chapter=chapter.strip(),
            topic=topic.strip(),
            tags=self._normalize_strings(
                tags
            ),
            metadata=dict(metadata or {}),
        )

        with self._lock:

            self._decks[
                deck.id
            ] = deck

            self._persist_deck(
                deck
            )

        self._emit(
            "education.flashcards.deck_created",
            deck.to_dict(),
        )

        return deck

    def get_deck(
        self,
        deck_id: str,
    ) -> FlashcardDeck | None:

        with self._lock:

            return self._decks.get(
                deck_id
            )

    def get_decks(
        self,
        *,
        include_archived: bool = False,
    ) -> list[FlashcardDeck]:

        with self._lock:

            decks = list(
                self._decks.values()
            )

        if not include_archived:

            decks = [
                deck
                for deck in decks
                if deck.status
                != "archived"
            ]

        return decks

    def archive_deck(
        self,
        deck_id: str,
    ) -> FlashcardDeck:

        with self._lock:

            deck = self._require_deck(
                deck_id
            )

            deck.status = "archived"

            deck.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_deck(
                deck
            )

        self._emit(
            "education.flashcards.deck_archived",
            deck.to_dict(),
        )

        return deck

    def restore_deck(
        self,
        deck_id: str,
    ) -> FlashcardDeck:

        with self._lock:

            deck = self._require_deck(
                deck_id
            )

            deck.status = "active"

            deck.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_deck(
                deck
            )

        return deck

    # ========================================================
    # CARD CREATION
    # ========================================================

    def create_card(
        self,
        *,
        deck_id: str,
        front: str,
        back: str,
        subject: str = "",
        chapter: str = "",
        topic: str = "",
        hint: str = "",
        explanation: str = "",
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Flashcard:

        front = self._required_text(
            front,
            "front",
        )

        back = self._required_text(
            back,
            "back",
        )

        with self._lock:

            deck = self._require_deck(
                deck_id
            )

            if deck.status == "archived":

                raise RuntimeError(
                    "Cannot add cards to "
                    "an archived deck."
                )

            card = Flashcard(
                id=self._new_card_id(),
                front=front,
                back=back,
                deck_id=deck_id,
                subject=(
                    subject.strip()
                    or deck.subject
                ),
                chapter=(
                    chapter.strip()
                    or deck.chapter
                ),
                topic=(
                    topic.strip()
                    or deck.topic
                ),
                hint=hint.strip(),
                explanation=explanation.strip(),
                tags=self._normalize_strings(
                    tags
                ),
                metadata=dict(metadata or {}),
            )

            self._cards[
                card.id
            ] = card

            if card.id not in deck.card_ids:

                deck.card_ids.append(
                    card.id
                )

            deck.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_card(
                card
            )

            self._persist_deck(
                deck
            )

        self._emit(
            "education.flashcards.card_created",
            card.to_dict(),
        )

        return card

    def get_card(
        self,
        card_id: str,
    ) -> Flashcard | None:

        with self._lock:

            return self._cards.get(
                card_id
            )

    def get_cards(
        self,
        deck_id: str | None = None,
    ) -> list[Flashcard]:

        with self._lock:

            cards = list(
                self._cards.values()
            )

        if deck_id:

            cards = [
                card
                for card in cards
                if card.deck_id
                == deck_id
            ]

        return cards

    def delete_card(
        self,
        card_id: str,
    ) -> bool:

        with self._lock:

            card = self._cards.pop(
                card_id,
                None,
            )

            if card is None:
                return False

            deck = self._decks.get(
                card.deck_id
            )

            if deck:

                if card_id in deck.card_ids:

                    deck.card_ids.remove(
                        card_id
                    )

                deck.updated_at = (
                    datetime.now().isoformat()
                )

                self._persist_deck(
                    deck
                )

        self._emit(
            "education.flashcards.card_deleted",
            {
                "card_id": card_id,
            },
        )

        return True

    # ========================================================
    # CARD EDITING
    # ========================================================

    def update_card(
        self,
        card_id: str,
        *,
        front: str | None = None,
        back: str | None = None,
        hint: str | None = None,
        explanation: str | None = None,
        topic: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> Flashcard:

        with self._lock:

            card = self._require_card(
                card_id
            )

            if front is not None:

                card.front = (
                    self._required_text(
                        front,
                        "front",
                    )
                )

            if back is not None:

                card.back = (
                    self._required_text(
                        back,
                        "back",
                    )
                )

            if hint is not None:
                card.hint = hint.strip()

            if explanation is not None:
                card.explanation = (
                    explanation.strip()
                )

            if topic is not None:
                card.topic = topic.strip()

            if tags is not None:
                card.tags = (
                    self._normalize_strings(
                        tags
                    )
                )

            self._persist_card(
                card
            )

        return card

    # ========================================================
    # REVIEW SYSTEM
    # ========================================================

    def review_card(
        self,
        card_id: str,
        rating: str,
        *,
        response_time_seconds: float | None = None,
    ) -> FlashcardReview:

        rating = (
            rating.strip().lower()
        )

        if rating not in RATINGS:

            raise ValueError(
                "rating must be one of: "
                + ", ".join(RATINGS)
            )

        if (
            response_time_seconds is not None
            and response_time_seconds < 0
        ):

            raise ValueError(
                "response_time_seconds "
                "cannot be negative."
            )

        with self._lock:

            card = self._require_card(
                card_id
            )

            previous_interval = (
                card.interval_days
            )

            previous_ease = (
                card.ease_factor
            )

            self._apply_rating(
                card,
                rating,
            )

            review = FlashcardReview(
                id=self._new_review_id(),
                card_id=card.id,
                rating=rating,
                previous_interval_days=(
                    previous_interval
                ),
                new_interval_days=(
                    card.interval_days
                ),
                previous_ease_factor=(
                    previous_ease
                ),
                new_ease_factor=(
                    card.ease_factor
                ),
                response_time_seconds=(
                    response_time_seconds
                ),
            )

            self._reviews[
                review.id
            ] = review

            self._persist_card(
                card
            )

            self._persist_review(
                review
            )

        self._emit(
            "education.flashcards.card_reviewed",
            {
                "card": card.to_dict(),
                "review": review.to_dict(),
            },
        )

        return review

    def _apply_rating(
        self,
        card: Flashcard,
        rating: str,
    ) -> None:

        now = datetime.now()

        card.last_reviewed_at = (
            now.isoformat()
        )

        if rating == "again":

            card.lapses += 1

            card.repetitions = 0

            card.interval_days = 0

            card.ease_factor = max(
                1.3,
                card.ease_factor - 0.20,
            )

            card.status = "learning"

            card.due_at = (
                now
                + timedelta(minutes=10)
            ).isoformat()

            return

        if rating == "hard":

            card.ease_factor = max(
                1.3,
                card.ease_factor - 0.15,
            )

            if card.repetitions == 0:

                card.interval_days = 1

            else:

                card.interval_days = max(
                    1,
                    round(
                        card.interval_days
                        * 1.2
                    ),
                )

            card.repetitions += 1

            card.status = "review"

        elif rating == "good":

            if card.repetitions == 0:

                card.interval_days = 1

            elif card.repetitions == 1:

                card.interval_days = 6

            else:

                card.interval_days = max(
                    1,
                    round(
                        card.interval_days
                        * card.ease_factor
                    ),
                )

            card.repetitions += 1

            card.status = "review"

        elif rating == "easy":

            card.ease_factor += 0.15

            if card.repetitions == 0:

                card.interval_days = 4

            else:

                card.interval_days = max(
                    2,
                    round(
                        card.interval_days
                        * card.ease_factor
                        * 1.3
                    ),
                )

            card.repetitions += 1

            card.status = "review"

        card.due_at = (
            now
            + timedelta(
                days=card.interval_days
            )
        ).isoformat()

    # ========================================================
    # DUE CARDS
    # ========================================================

    def get_due_cards(
        self,
        deck_id: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[Flashcard]:

        now = datetime.now()

        cards = self.get_cards(
            deck_id
        )

        due = []

        for card in cards:

            if card.status == "suspended":
                continue

            try:

                due_at = datetime.fromisoformat(
                    card.due_at
                )

            except ValueError:

                due_at = now

            if due_at <= now:

                due.append(card)

        due.sort(
            key=lambda card: card.due_at
        )

        if limit is not None:

            if limit <= 0:
                return []

            due = due[:limit]

        return due

    # ========================================================
    # STUDY QUEUE
    # ========================================================

    def build_study_queue(
        self,
        deck_id: str,
        *,
        limit: int = 20,
    ) -> list[Flashcard]:

        if limit <= 0:

            raise ValueError(
                "limit must be greater than zero."
            )

        cards = self.get_cards(
            deck_id
        )

        due_cards = []

        new_cards = []

        learning_cards = []

        for card in cards:

            if card.status == "suspended":
                continue

            if card.status == "new":

                new_cards.append(card)

                continue

            if card.status == "learning":

                learning_cards.append(card)

                continue

            try:

                due_at = datetime.fromisoformat(
                    card.due_at
                )

            except ValueError:

                due_at = datetime.now()

            if due_at <= datetime.now():

                due_cards.append(card)

        queue = (
            due_cards
            + learning_cards
            + new_cards
        )

        return queue[:limit]

    # ========================================================
    # SUSPEND / UNSUSPEND
    # ========================================================

    def suspend_card(
        self,
        card_id: str,
    ) -> Flashcard:

        with self._lock:

            card = self._require_card(
                card_id
            )

            card.status = "suspended"

            self._persist_card(
                card
            )

        self._emit(
            "education.flashcards.card_suspended",
            card.to_dict(),
        )

        return card

    def unsuspend_card(
        self,
        card_id: str,
    ) -> Flashcard:

        with self._lock:

            card = self._require_card(
                card_id
            )

            if card.repetitions == 0:

                card.status = "new"

            else:

                card.status = "review"

            self._persist_card(
                card
            )

        return card

    # ========================================================
    # RESET CARD
    # ========================================================

    def reset_card(
        self,
        card_id: str,
    ) -> Flashcard:

        with self._lock:

            card = self._require_card(
                card_id
            )

            card.status = "new"

            card.difficulty = 0.0

            card.repetitions = 0

            card.lapses = 0

            card.ease_factor = 2.5

            card.interval_days = 0

            card.due_at = (
                datetime.now().isoformat()
            )

            card.last_reviewed_at = None

            self._persist_card(
                card
            )

        return card

    # ========================================================
    # SEARCH
    # ========================================================

    def search_cards(
        self,
        query: str,
    ) -> list[Flashcard]:

        query = self._required_text(
            query,
            "query",
        ).casefold()

        cards = self.get_cards()

        results = []

        for card in cards:

            searchable = " ".join(
                [
                    card.front,
                    card.back,
                    card.hint,
                    card.explanation,
                    card.subject,
                    card.chapter,
                    card.topic,
                    " ".join(card.tags),
                ]
            ).casefold()

            if query in searchable:

                results.append(card)

        return results

    def get_cards_by_topic(
        self,
        topic: str,
    ) -> list[Flashcard]:

        topic = self._required_text(
            topic,
            "topic",
        ).casefold()

        return [
            card
            for card in self.get_cards()
            if card.topic.casefold()
            == topic
        ]

    def get_cards_by_subject(
        self,
        subject: str,
    ) -> list[Flashcard]:

        subject = self._required_text(
            subject,
            "subject",
        ).casefold()

        return [
            card
            for card in self.get_cards()
            if card.subject.casefold()
            == subject
        ]

    # ========================================================
    # WEAK CARDS
    # ========================================================

    def get_weak_cards(
        self,
        *,
        minimum_reviews: int = 2,
    ) -> list[dict[str, Any]]:

        if minimum_reviews <= 0:

            raise ValueError(
                "minimum_reviews must be "
                "greater than zero."
            )

        with self._lock:

            cards = dict(
                self._cards
            )

            reviews = list(
                self._reviews.values()
            )

        grouped: dict[
            str,
            list[FlashcardReview],
        ] = {}

        for review in reviews:

            grouped.setdefault(
                review.card_id,
                [],
            ).append(review)

        results = []

        for card_id, card_reviews in grouped.items():

            if len(card_reviews) < minimum_reviews:
                continue

            failures = sum(
                review.rating == "again"
                for review in card_reviews
            )

            hard = sum(
                review.rating == "hard"
                for review in card_reviews
            )

            total = len(card_reviews)

            difficulty_score = (
                (failures * 2)
                + hard
            ) / total

            card = cards.get(
                card_id
            )

            if card is None:
                continue

            results.append(
                {
                    "card_id": card.id,
                    "front": card.front,
                    "topic": card.topic,
                    "reviews": total,
                    "again": failures,
                    "hard": hard,
                    "difficulty_score": round(
                        difficulty_score,
                        3,
                    ),
                }
            )

        results.sort(
            key=lambda item: (
                -item["difficulty_score"],
                -item["reviews"],
            )
        )

        return results

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(
        self,
        deck_id: str | None = None,
    ) -> FlashcardStatistics:

        decks = self.get_decks(
            include_archived=True
        )

        cards = self.get_cards(
            deck_id
        )

        if deck_id:

            decks = [
                deck
                for deck in decks
                if deck.id == deck_id
            ]

        with self._lock:

            reviews = list(
                self._reviews.values()
            )

        if deck_id:

            card_ids = {
                card.id
                for card in cards
            }

            reviews = [
                review
                for review in reviews
                if review.card_id
                in card_ids
            ]

        stats = FlashcardStatistics()

        stats.total_decks = len(
            decks
        )

        stats.total_cards = len(
            cards
        )

        for card in cards:

            if card.status == "new":

                stats.new_cards += 1

            elif card.status == "learning":

                stats.learning_cards += 1

            elif card.status == "review":

                stats.review_cards += 1

            elif card.status == "suspended":

                stats.suspended_cards += 1

        stats.due_cards = len(
            self.get_due_cards(
                deck_id
            )
        )

        stats.total_reviews = len(
            reviews
        )

        stats.correct_reviews = sum(
            review.rating
            in {"good", "easy"}
            for review in reviews
        )

        if cards:

            stats.average_interval_days = (
                sum(
                    card.interval_days
                    for card in cards
                )
                / len(cards)
            )

            stats.average_ease_factor = (
                sum(
                    card.ease_factor
                    for card in cards
                )
                / len(cards)
            )

        return stats

    # ========================================================
    # EXPORT / IMPORT
    # ========================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "decks": [
                    deck.to_dict()
                    for deck
                    in self._decks.values()
                ],
                "cards": [
                    card.to_dict()
                    for card
                    in self._cards.values()
                ],
                "reviews": [
                    review.to_dict()
                    for review
                    in self._reviews.values()
                ],
            }

    def import_state(
        self,
        data: dict[str, Any],
        *,
        replace: bool = False,
    ) -> None:

        if not isinstance(
            data,
            dict,
        ):

            raise TypeError(
                "Flashcard state must "
                "be a dictionary."
            )

        with self._lock:

            if replace:

                self._decks.clear()
                self._cards.clear()
                self._reviews.clear()

            for raw in data.get(
                "decks",
                [],
            ):

                deck = FlashcardDeck(
                    **raw
                )

                self._decks[
                    deck.id
                ] = deck

            for raw in data.get(
                "cards",
                [],
            ):

                card = Flashcard(
                    **raw
                )

                self._cards[
                    card.id
                ] = card

            for raw in data.get(
                "reviews",
                [],
            ):

                review = FlashcardReview(
                    **raw
                )

                self._reviews[
                    review.id
                ] = review

    # ========================================================
    # STORAGE
    # ========================================================

    def _persist_deck(
        self,
        deck: FlashcardDeck,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_flashcard_deck",
            None,
        )

        if callable(method):

            try:

                method(
                    deck.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to persist flashcard deck."
                )

    def _persist_card(
        self,
        card: Flashcard,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_flashcard",
            None,
        )

        if callable(method):

            try:

                method(
                    card.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to persist flashcard."
                )

    def _persist_review(
        self,
        review: FlashcardReview,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_flashcard_review",
            None,
        )

        if callable(method):

            try:

                method(
                    review.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to persist flashcard review."
                )

    # ========================================================
    # EVENTS
    # ========================================================

    def _emit(
        self,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:

        if self.event_bus is None:
            return

        try:

            publish = getattr(
                self.event_bus,
                "publish",
                None,
            )

            if callable(publish):

                publish(
                    event_name,
                    payload,
                )

        except Exception:

            logger.exception(
                "Failed to publish event: %s",
                event_name,
            )

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    def _require_deck(
        self,
        deck_id: str,
    ) -> FlashcardDeck:

        deck = self._decks.get(
            deck_id
        )

        if deck is None:

            raise KeyError(
                f"Flashcard deck not found: "
                f"{deck_id}"
            )

        return deck

    def _require_card(
        self,
        card_id: str,
    ) -> Flashcard:

        card = self._cards.get(
            card_id
        )

        if card is None:

            raise KeyError(
                f"Flashcard not found: "
                f"{card_id}"
            )

        return card

    @staticmethod
    def _required_text(
        value: str,
        name: str,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):

            raise TypeError(
                f"{name} must be a string."
            )

        value = value.strip()

        if not value:

            raise ValueError(
                f"{name} cannot be empty."
            )

        return value

    @staticmethod
    def _normalize_strings(
        values: Iterable[str] | None,
    ) -> list[str]:

        if values is None:
            return []

        result = []

        for value in values:

            value = str(value).strip()

            if value and value not in result:

                result.append(value)

        return result

    @staticmethod
    def _new_deck_id() -> str:

        return (
            "deck_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _new_card_id() -> str:

        return (
            "card_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _new_review_id() -> str:

        return (
            "review_"
            + uuid.uuid4().hex
        )

    def __len__(self) -> int:

        with self._lock:

            return len(
                self._cards
            )


__all__ = [
    "RATINGS",
    "CARD_STATUSES",
    "DECK_STATUSES",
    "Flashcard",
    "FlashcardDeck",
    "FlashcardReview",
    "FlashcardStatistics",
    "FlashcardEngine",
    "FlashcardManager",
]


FlashcardManager = FlashcardEngine
