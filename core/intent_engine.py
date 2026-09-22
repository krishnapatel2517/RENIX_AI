"""
RENIX AI - Intent Engine
========================

The Intent Engine converts raw human input into a structured intent.

It supports input from:
    - Voice
    - Text
    - Gestures
    - Holographic UI
    - Automation
    - Other RENIX components

Responsibilities:
    - Normalize input
    - Detect intent
    - Extract entities
    - Extract parameters
    - Detect urgency
    - Detect confirmation requirements
    - Detect command category
    - Handle aliases
    - Score possible intents
    - Allow external AI/LLM providers
    - Maintain intent history

The Intent Engine does NOT execute commands.
It only understands WHAT the user wants.
"""

from __future__ import annotations

import logging
import re
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(
    "RENIX.IntentEngine"
)


# ============================================================================
# ENUMS
# ============================================================================

class IntentConfidence(str, Enum):
    """
    Confidence classification for detected intents.
    """

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class IntentSource(str, Enum):
    """
    Source of the input.
    """

    VOICE = "voice"
    TEXT = "text"
    GESTURE = "gesture"
    UI = "ui"
    AUTOMATION = "automation"
    SYSTEM = "system"
    UNKNOWN = "unknown"


# ============================================================================
# INTENT
# ============================================================================

@dataclass
class RENIXIntent:
    """
    Structured representation of a detected intent.
    """

    intent_id: str

    name: str

    raw_input: str = ""

    normalized_input: str = ""

    confidence: float = 0.0

    confidence_level: IntentConfidence = (
        IntentConfidence.VERY_LOW
    )

    source: IntentSource = (
        IntentSource.UNKNOWN
    )

    category: str = "general"

    entities: dict[str, Any] = field(
        default_factory=dict
    )

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    keywords: list[str] = field(
        default_factory=list
    )

    alternatives: list[dict[str, Any]] = field(
        default_factory=list
    )

    requires_confirmation: bool = False

    urgency: int = 0

    destructive: bool = False

    created_at: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert intent to a serializable dictionary.
        """

        return {
            "intent_id": self.intent_id,
            "name": self.name,
            "raw_input": self.raw_input,
            "normalized_input": self.normalized_input,
            "confidence": self.confidence,
            "confidence_level": (
                self.confidence_level.value
            ),
            "source": self.source.value,
            "category": self.category,
            "entities": dict(
                self.entities
            ),
            "parameters": dict(
                self.parameters
            ),
            "keywords": list(
                self.keywords
            ),
            "alternatives": list(
                self.alternatives
            ),
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "urgency": self.urgency,
            "destructive": self.destructive,
            "created_at": self.created_at,
            "metadata": dict(
                self.metadata
            ),
        }


# ============================================================================
# INTENT RULE
# ============================================================================

@dataclass
class IntentRule:
    """
    Rule used to recognize an intent.
    """

    name: str

    keywords: list[str] = field(
        default_factory=list
    )

    phrases: list[str] = field(
        default_factory=list
    )

    category: str = "general"

    priority: int = 0

    confidence_boost: float = 0.0

    entities: dict[str, Any] = field(
        default_factory=dict
    )

    requires_confirmation: bool = False

    destructive: bool = False

    enabled: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "name": self.name,
            "keywords": list(
                self.keywords
            ),
            "phrases": list(
                self.phrases
            ),
            "category": self.category,
            "priority": self.priority,
            "confidence_boost": (
                self.confidence_boost
            ),
            "entities": dict(
                self.entities
            ),
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "destructive": self.destructive,
            "enabled": self.enabled,
            "metadata": dict(
                self.metadata
            ),
        }


# ============================================================================
# INTENT ENGINE
# ============================================================================

class IntentEngine:
    """
    RENIX Intent Detection Engine.
    """

    def __init__(
        self,
        *,
        max_history: int = 500,
        minimum_confidence: float = 0.20,
    ) -> None:

        self.logger = logger

        self.max_history = max(
            1,
            int(max_history),
        )

        self.minimum_confidence = max(
            0.0,
            min(
                1.0,
                float(
                    minimum_confidence
                ),
            ),
        )

        # --------------------------------------------------------------------
        # Rules
        # --------------------------------------------------------------------

        self.rules: dict[
            str,
            IntentRule,
        ] = {}

        # --------------------------------------------------------------------
        # Custom detectors
        # --------------------------------------------------------------------

        self.detectors: list[
            Callable[..., Any]
        ] = []

        # --------------------------------------------------------------------
        # External AI detector
        # --------------------------------------------------------------------

        self.ai_detector: Optional[
            Callable[..., Any]
        ] = None

        # --------------------------------------------------------------------
        # History
        # --------------------------------------------------------------------

        self.history: list[
            RENIXIntent
        ] = []

        # --------------------------------------------------------------------
        # Counters
        # --------------------------------------------------------------------

        self._intent_counter = 0

        self.total_processed = 0

        self.total_detected = 0

        self.total_unknown = 0

        # --------------------------------------------------------------------
        # Built-in rules
        # --------------------------------------------------------------------

        self._register_builtin_rules()

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the intent engine.
        """

        self.logger.info(
            "RENIX Intent Engine initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the intent engine.
        """

        self.rules.clear()

        self.detectors.clear()

        self.ai_detector = None

        self.logger.info(
            "RENIX Intent Engine shutdown."
        )

    # ========================================================================
    # ID GENERATION
    # ========================================================================

    def _generate_intent_id(
        self,
    ) -> str:
        """
        Generate unique intent ID.
        """

        self._intent_counter += 1

        return (
            f"renix_intent_"
            f"{int(time.time() * 1000)}_"
            f"{self._intent_counter}"
        )

    # ========================================================================
    # NORMALIZE INPUT
    # ========================================================================

    @staticmethod
    def normalize_input(
        text: str,
    ) -> str:
        """
        Normalize human input.

        Examples:

            "  Open   Chrome!!! "
                ->
            "open chrome"

        """

        if not text:

            return ""

        text = str(text)

        text = text.strip().lower()

        text = re.sub(
            r"[^\w\s:/\\.-@%+]",
            " ",
            text,
            flags=re.UNICODE,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ========================================================================
    # TOKENIZE
    # ========================================================================

    @staticmethod
    def tokenize(
        text: str,
    ) -> list[str]:
        """
        Split normalized text into useful tokens.
        """

        if not text:

            return []

        return [
            token
            for token in text.split()
            if token.strip()
        ]

    # ========================================================================
    # REGISTER RULE
    # ========================================================================

    def register_rule(
        self,
        rule: IntentRule,
    ) -> IntentRule:
        """
        Register an intent recognition rule.
        """

        if not isinstance(
            rule,
            IntentRule,
        ):

            raise TypeError(
                "rule must be an IntentRule."
            )

        name = (
            rule.name.strip().lower()
        )

        if not name:

            raise ValueError(
                "Intent rule name cannot be empty."
            )

        rule.name = name

        rule.keywords = [
            self.normalize_input(
                keyword
            )
            for keyword in rule.keywords
            if keyword
        ]

        rule.phrases = [
            self.normalize_input(
                phrase
            )
            for phrase in rule.phrases
            if phrase
        ]

        self.rules[name] = rule

        return rule

    # ========================================================================
    # REGISTER SIMPLE RULE
    # ========================================================================

    def register(
        self,
        name: str,
        *,
        keywords: Optional[
            list[str]
        ] = None,
        phrases: Optional[
            list[str]
        ] = None,
        category: str = "general",
        priority: int = 0,
        confidence_boost: float = 0.0,
        entities: Optional[
            dict[str, Any]
        ] = None,
        requires_confirmation: bool = False,
        destructive: bool = False,
    ) -> IntentRule:
        """
        Convenience method for registering an intent rule.
        """

        rule = IntentRule(
            name=name,
            keywords=(
                keywords or []
            ),
            phrases=(
                phrases or []
            ),
            category=category,
            priority=priority,
            confidence_boost=confidence_boost,
            entities=(
                entities or {}
            ),
            requires_confirmation=(
                requires_confirmation
            ),
            destructive=destructive,
        )

        return self.register_rule(
            rule
        )

    # ========================================================================
    # REMOVE RULE
    # ========================================================================

    def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Remove an intent rule.
        """

        name = self.normalize_input(
            name
        )

        return (
            self.rules.pop(
                name,
                None,
            )
            is not None
        )

    # ========================================================================
    # ADD DETECTOR
    # ========================================================================

    def add_detector(
        self,
        detector: Callable[..., Any],
    ) -> None:
        """
        Add a custom intent detector.

        Detector can return:

            RENIXIntent

        or:

            dict
        """

        if not callable(
            detector
        ):

            raise TypeError(
                "Detector must be callable."
            )

        self.detectors.append(
            detector
        )

    # ========================================================================
    # SET AI DETECTOR
    # ========================================================================

    def set_ai_detector(
        self,
        detector: Optional[
            Callable[..., Any]
        ],
    ) -> None:
        """
        Set an external AI/LLM intent detector.

        The external detector is expected to return
        either RENIXIntent or a dictionary.
        """

        if (
            detector is not None
            and not callable(detector)
        ):

            raise TypeError(
                "AI detector must be callable."
            )

        self.ai_detector = detector

    # ========================================================================
    # BUILT-IN RULES
    # ========================================================================

    def _register_builtin_rules(
        self,
    ) -> None:
        """
        Register common RENIX intents.
        """

        builtins = [

            # ---------------------------------------------------------------
            # Computer
            # ---------------------------------------------------------------

            IntentRule(
                name="open_application",
                keywords=[
                    "open",
                    "launch",
                    "start",
                    "run",
                ],
                phrases=[
                    "open",
                    "launch",
                    "start",
                    "run",
                ],
                category="computer",
                priority=5,
            ),

            IntentRule(
                name="close_application",
                keywords=[
                    "close",
                    "exit",
                    "quit",
                    "stop",
                ],
                phrases=[
                    "close",
                    "exit",
                    "quit",
                ],
                category="computer",
                priority=5,
            ),

            IntentRule(
                name="minimize_window",
                keywords=[
                    "minimize",
                ],
                phrases=[
                    "minimize window",
                    "minimise window",
                ],
                category="computer",
                priority=5,
            ),

            IntentRule(
                name="maximize_window",
                keywords=[
                    "maximize",
                    "maximise",
                ],
                phrases=[
                    "maximize window",
                    "maximise window",
                ],
                category="computer",
                priority=5,
            ),

            IntentRule(
                name="switch_window",
                keywords=[
                    "switch",
                ],
                phrases=[
                    "switch window",
                    "switch application",
                    "change window",
                ],
                category="computer",
                priority=4,
            ),

            # ---------------------------------------------------------------
            # Files
            # ---------------------------------------------------------------

            IntentRule(
                name="find_file",
                keywords=[
                    "find",
                    "search",
                    "locate",
                ],
                phrases=[
                    "find file",
                    "search file",
                    "locate file",
                ],
                category="files",
                priority=5,
            ),

            IntentRule(
                name="copy_file",
                keywords=[
                    "copy",
                ],
                phrases=[
                    "copy file",
                    "copy this",
                    "make a copy",
                ],
                category="files",
                priority=5,
            ),

            IntentRule(
                name="move_file",
                keywords=[
                    "move",
                    "shift",
                ],
                phrases=[
                    "move file",
                    "move this file",
                    "shift file",
                ],
                category="files",
                priority=5,
            ),

            IntentRule(
                name="rename_file",
                keywords=[
                    "rename",
                ],
                phrases=[
                    "rename file",
                    "rename this",
                ],
                category="files",
                priority=5,
            ),

            IntentRule(
                name="delete_file",
                keywords=[
                    "delete",
                    "remove",
                ],
                phrases=[
                    "delete file",
                    "delete this",
                    "remove file",
                ],
                category="files",
                priority=6,
                requires_confirmation=True,
                destructive=True,
            ),

            # ---------------------------------------------------------------
            # Browser
            # ---------------------------------------------------------------

            IntentRule(
                name="web_search",
                keywords=[
                    "search",
                    "google",
                    "look",
                    "find",
                ],
                phrases=[
                    "search the web",
                    "search online",
                    "google",
                    "look this up",
                ],
                category="browser",
                priority=4,
            ),

            IntentRule(
                name="open_website",
                keywords=[
                    "website",
                    "site",
                    "url",
                ],
                phrases=[
                    "open website",
                    "open site",
                    "go to website",
                ],
                category="browser",
                priority=4,
            ),

            # ---------------------------------------------------------------
            # Media
            # ---------------------------------------------------------------

            IntentRule(
                name="play_music",
                keywords=[
                    "play",
                    "music",
                    "song",
                ],
                phrases=[
                    "play music",
                    "play song",
                    "play a song",
                ],
                category="media",
                priority=4,
            ),

            IntentRule(
                name="pause_media",
                keywords=[
                    "pause",
                ],
                phrases=[
                    "pause",
                    "pause music",
                    "pause video",
                ],
                category="media",
                priority=5,
            ),

            IntentRule(
                name="resume_media",
                keywords=[
                    "resume",
                    "continue",
                ],
                phrases=[
                    "resume",
                    "resume music",
                    "continue playing",
                ],
                category="media",
                priority=5,
            ),

            IntentRule(
                name="increase_volume",
                keywords=[
                    "increase",
                    "volume",
                    "louder",
                ],
                phrases=[
                    "increase volume",
                    "make it louder",
                ],
                category="media",
                priority=5,
            ),

            IntentRule(
                name="decrease_volume",
                keywords=[
                    "decrease",
                    "volume",
                    "quieter",
                ],
                phrases=[
                    "decrease volume",
                    "make it quieter",
                ],
                category="media",
                priority=5,
            ),

            # ---------------------------------------------------------------
            # System
            # ---------------------------------------------------------------

            IntentRule(
                name="system_status",
                keywords=[
                    "status",
                    "system",
                    "cpu",
                    "ram",
                    "memory",
                ],
                phrases=[
                    "system status",
                    "check system",
                    "system health",
                ],
                category="system",
                priority=4,
            ),

            IntentRule(
                name="shutdown_system",
                keywords=[
                    "shutdown",
                    "shut",
                ],
                phrases=[
                    "shutdown computer",
                    "shut down computer",
                ],
                category="system",
                priority=10,
                requires_confirmation=True,
                destructive=True,
            ),

            IntentRule(
                name="restart_system",
                keywords=[
                    "restart",
                    "reboot",
                ],
                phrases=[
                    "restart computer",
                    "reboot computer",
                ],
                category="system",
                priority=9,
                requires_confirmation=True,
            ),

            # ---------------------------------------------------------------
            # Communication
            # ---------------------------------------------------------------

            IntentRule(
                name="send_message",
                keywords=[
                    "send",
                    "message",
                    "text",
                ],
                phrases=[
                    "send message",
                    "send a message",
                    "text someone",
                ],
                category="communication",
                priority=5,
            ),

            IntentRule(
                name="make_call",
                keywords=[
                    "call",
                    "phone",
                    "dial",
                ],
                phrases=[
                    "make a call",
                    "call someone",
                    "phone someone",
                ],
                category="communication",
                priority=5,
            ),

            # ---------------------------------------------------------------
            # Education
            # ---------------------------------------------------------------

            IntentRule(
                name="study",
                keywords=[
                    "study",
                    "learn",
                    "revise",
                ],
                phrases=[
                    "help me study",
                    "start studying",
                    "study this",
                ],
                category="education",
                priority=4,
            ),

            IntentRule(
                name="create_quiz",
                keywords=[
                    "quiz",
                    "questions",
                ],
                phrases=[
                    "make a quiz",
                    "create quiz",
                    "quiz me",
                ],
                category="education",
                priority=5,
            ),

            IntentRule(
                name="create_timetable",
                keywords=[
                    "timetable",
                    "schedule",
                ],
                phrases=[
                    "make timetable",
                    "create timetable",
                    "make a study schedule",
                ],
                category="education",
                priority=5,
            ),

            # ---------------------------------------------------------------
            # Personal
            # ---------------------------------------------------------------

            IntentRule(
                name="create_reminder",
                keywords=[
                    "remind",
                    "reminder",
                ],
                phrases=[
                    "remind me",
                    "set reminder",
                    "create reminder",
                ],
                category="personal",
                priority=5,
            ),

            IntentRule(
                name="create_note",
                keywords=[
                    "note",
                    "remember",
                    "write",
                ],
                phrases=[
                    "take a note",
                    "write a note",
                    "make a note",
                ],
                category="personal",
                priority=4,
            ),

            IntentRule(
                name="set_timer",
                keywords=[
                    "timer",
                    "countdown",
                ],
                phrases=[
                    "set timer",
                    "start timer",
                    "set a timer",
                ],
                category="personal",
                priority=5,
            ),

            IntentRule(
                name="set_alarm",
                keywords=[
                    "alarm",
                ],
                phrases=[
                    "set alarm",
                    "wake me",
                    "wake me up",
                ],
                category="personal",
                priority=5,
            ),

            # ---------------------------------------------------------------
            # Coding
            # ---------------------------------------------------------------

            IntentRule(
                name="write_code",
                keywords=[
                    "code",
                    "program",
                    "script",
                ],
                phrases=[
                    "write code",
                    "create code",
                    "write a program",
                ],
                category="coding",
                priority=5,
            ),

            IntentRule(
                name="debug_code",
                keywords=[
                    "debug",
                    "error",
                    "fix",
                ],
                phrases=[
                    "debug this",
                    "fix this error",
                    "find the error",
                ],
                category="coding",
                priority=6,
            ),

            IntentRule(
                name="run_code",
                keywords=[
                    "run",
                    "execute",
                ],
                phrases=[
                    "run code",
                    "execute code",
                    "run the program",
                ],
                category="coding",
                priority=5,
            ),

            # ---------------------------------------------------------------
            # Research
            # ---------------------------------------------------------------

            IntentRule(
                name="research_topic",
                keywords=[
                    "research",
                    "investigate",
                    "analyze",
                ],
                phrases=[
                    "research this",
                    "research topic",
                    "investigate this",
                ],
                category="research",
                priority=6,
            ),

            IntentRule(
                name="compare_items",
                keywords=[
                    "compare",
                    "difference",
                ],
                phrases=[
                    "compare these",
                    "compare them",
                    "what is the difference",
                ],
                category="research",
                priority=5,
            ),

            # ---------------------------------------------------------------
            # Automation
            # ---------------------------------------------------------------

            IntentRule(
                name="create_automation",
                keywords=[
                    "automate",
                    "automation",
                    "workflow",
                ],
                phrases=[
                    "automate this",
                    "create automation",
                    "create workflow",
                ],
                category="automation",
                priority=6,
            ),

            # ---------------------------------------------------------------
            # Greeting / conversation
            # ---------------------------------------------------------------

            IntentRule(
                name="greeting",
                keywords=[
                    "hello",
                    "hi",
                    "hey",
                    "hii",
                    "hiii",
                ],
                phrases=[
                    "hello",
                    "hi",
                    "hey",
                    "good morning",
                    "good afternoon",
                    "good evening",
                ],
                category="conversation",
                priority=3,
            ),

            IntentRule(
                name="goodbye",
                keywords=[
                    "bye",
                    "goodbye",
                ],
                phrases=[
                    "goodbye",
                    "good bye",
                    "see you",
                    "bye",
                ],
                category="conversation",
                priority=3,
            ),

            IntentRule(
                name="help",
                keywords=[
                    "help",
                    "assist",
                ],
                phrases=[
                    "help me",
                    "what can you do",
                    "how can you help",
                ],
                category="conversation",
                priority=3,
            ),
        ]

        for rule in builtins:

            self.register_rule(
                rule
            )

    # ========================================================================
    # SCORE RULE
    # ========================================================================

    def _score_rule(
        self,
        rule: IntentRule,
        normalized_text: str,
        tokens: list[str],
    ) -> float:
        """
        Calculate confidence score for a rule.
        """

        if not rule.enabled:

            return 0.0

        score = 0.0

        # --------------------------------------------------------------------
        # Exact phrase matches
        # --------------------------------------------------------------------

        for phrase in rule.phrases:

            if not phrase:

                continue

            if phrase in normalized_text:

                score += 0.55

        # --------------------------------------------------------------------
        # Keyword matches
        # --------------------------------------------------------------------

        matched_keywords = 0

        for keyword in rule.keywords:

            if not keyword:

                continue

            if keyword in tokens:

                matched_keywords += 1

        if rule.keywords:

            keyword_ratio = (
                matched_keywords
                / len(rule.keywords)
            )

            score += (
                keyword_ratio
                * 0.40
            )

        # --------------------------------------------------------------------
        # Priority
        # --------------------------------------------------------------------

        score += min(
            0.10,
            max(
                0.0,
                rule.priority
                * 0.01,
            ),
        )

        # --------------------------------------------------------------------
        # Custom boost
        # --------------------------------------------------------------------

        score += rule.confidence_boost

        return min(
            1.0,
            max(
                0.0,
                score,
            ),
        )

    # ========================================================================
    # CONFIDENCE LEVEL
    # ========================================================================

    @staticmethod
    def _confidence_level(
        confidence: float,
    ) -> IntentConfidence:
        """
        Convert numeric confidence into a category.
        """

        if confidence >= 0.90:

            return (
                IntentConfidence.VERY_HIGH
            )

        if confidence >= 0.75:

            return (
                IntentConfidence.HIGH
            )

        if confidence >= 0.50:

            return (
                IntentConfidence.MEDIUM
            )

        if confidence >= 0.25:

            return (
                IntentConfidence.LOW
            )

        return (
            IntentConfidence.VERY_LOW
        )

    # ========================================================================
    # EXTRACT ENTITIES
    # ========================================================================

    def extract_entities(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        Extract common entities from human input.

        This is intentionally generic so later AI/LLM
        components can enrich the result.
        """

        entities: dict[
            str,
            Any
        ] = {}

        # --------------------------------------------------------------------
        # URLs
        # --------------------------------------------------------------------

        urls = re.findall(
            r"(https?://[^\s]+|www\.[^\s]+)",
            text,
            flags=re.IGNORECASE,
        )

        if urls:

            entities["urls"] = urls

        # --------------------------------------------------------------------
        # Email addresses
        # --------------------------------------------------------------------

        emails = re.findall(
            r"\b[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text,
        )

        if emails:

            entities["emails"] = emails

        # --------------------------------------------------------------------
        # File extensions
        # --------------------------------------------------------------------

        extensions = re.findall(
            r"\b[\w.-]+\.(?:pdf|docx?|xlsx?|pptx?|"
            r"txt|csv|json|py|js|ts|html|css|png|"
            r"jpg|jpeg|gif|mp4|mp3|zip)\b",
            text,
            flags=re.IGNORECASE,
        )

        if extensions:

            entities["files"] = extensions

        # --------------------------------------------------------------------
        # Numbers
        # --------------------------------------------------------------------

        numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            text,
        )

        if numbers:

            entities["numbers"] = [
                float(number)
                if "." in number
                else int(number)
                for number in numbers
            ]

        # --------------------------------------------------------------------
        # Time
        # --------------------------------------------------------------------

        times = re.findall(
            r"\b\d{1,2}:\d{2}\s?"
            r"(?:am|pm)?\b",
            text,
            flags=re.IGNORECASE,
        )

        if times:

            entities["times"] = times

        # --------------------------------------------------------------------
        # Percentages
        # --------------------------------------------------------------------

        percentages = re.findall(
            r"\b\d+(?:\.\d+)?%",
            text,
        )

        if percentages:

            entities["percentages"] = percentages

        # --------------------------------------------------------------------
        # Quoted text
        # --------------------------------------------------------------------

        quoted = re.findall(
            r'"([^"]+)"|\'([^\']+)\'',
            text,
        )

        quoted_values = [
            first or second
            for first, second
            in quoted
        ]

        if quoted_values:

            entities["quoted"] = (
                quoted_values
            )

        return entities

    # ========================================================================
    # EXTRACT PARAMETERS
    # ========================================================================

    def extract_parameters(
        self,
        text: str,
        intent_name: str,
    ) -> dict[str, Any]:
        """
        Extract intent-specific parameters.
        """

        parameters: dict[
            str,
            Any
        ] = {}

        normalized = (
            self.normalize_input(
                text
            )
        )

        # --------------------------------------------------------------------
        # Application
        # --------------------------------------------------------------------

        if intent_name in {
            "open_application",
            "close_application",
        }:

            patterns = [
                r"(?:open|launch|start|run)"
                r"\s+(.+)",

                r"(?:close|exit|quit|stop)"
                r"\s+(.+)",
            ]

            for pattern in patterns:

                match = re.search(
                    pattern,
                    normalized,
                    flags=re.IGNORECASE,
                )

                if match:

                    parameters[
                        "application"
                    ] = match.group(
                        1
                    ).strip()

                    break

        # --------------------------------------------------------------------
        # File
        # --------------------------------------------------------------------

        if intent_name in {
            "find_file",
            "copy_file",
            "move_file",
            "rename_file",
            "delete_file",
        }:

            files = re.findall(
                r"\b[\w.- ]+\."
                r"(?:pdf|docx?|xlsx?|pptx?|"
                r"txt|csv|json|py|js|ts|"
                r"png|jpg|jpeg|zip)\b",
                normalized,
                flags=re.IGNORECASE,
            )

            if files:

                parameters[
                    "file"
                ] = files[0].strip()

        # --------------------------------------------------------------------
        # Search query
        # --------------------------------------------------------------------

        if intent_name in {
            "web_search",
            "research_topic",
            "compare_items",
        }:

            patterns = [
                r"(?:search|research|investigate)"
                r"\s+(?:for\s+)?(.+)",

                r"compare\s+(.+)",
            ]

            for pattern in patterns:

                match = re.search(
                    pattern,
                    normalized,
                    flags=re.IGNORECASE,
                )

                if match:

                    parameters[
                        "query"
                    ] = match.group(
                        1
                    ).strip()

                    break

        # --------------------------------------------------------------------
        # Website
        # --------------------------------------------------------------------

        if intent_name == "open_website":

            urls = re.findall(
                r"(https?://[^\s]+|"
                r"www\.[^\s]+|"
                r"[a-zA-Z0-9-]+\."
                r"(?:com|org|net|in|io|ai))",
                normalized,
                flags=re.IGNORECASE,
            )

            if urls:

                parameters[
                    "url"
                ] = urls[0]

        # --------------------------------------------------------------------
        # Timer
        # --------------------------------------------------------------------

        if intent_name == "set_timer":

            duration_match = re.search(
                r"(\d+(?:\.\d+)?)\s*"
                r"(seconds?|secs?|"
                r"minutes?|mins?|"
                r"hours?|hrs?)",
                normalized,
                flags=re.IGNORECASE,
            )

            if duration_match:

                value = float(
                    duration_match.group(
                        1
                    )
                )

                unit = (
                    duration_match.group(
                        2
                    ).lower()
                )

                if unit.startswith(
                    "hour"
                ) or unit.startswith(
                    "hr"
                ):

                    seconds = value * 3600

                elif unit.startswith(
                    "min"
                ):

                    seconds = value * 60

                else:

                    seconds = value

                parameters[
                    "duration_seconds"
                ] = int(
                    seconds
                )

        # --------------------------------------------------------------------
        # Message
        # --------------------------------------------------------------------

        if intent_name == "send_message":

            match = re.search(
                r"(?:send|text)"
                r"\s+(?:a\s+)?message"
                r"\s*(?:to\s+)?"
                r"(.+)",
                normalized,
                flags=re.IGNORECASE,
            )

            if match:

                parameters[
                    "message_target"
                ] = match.group(
                    1
                ).strip()

        # --------------------------------------------------------------------
        # Music
        # --------------------------------------------------------------------

        if intent_name == "play_music":

            match = re.search(
                r"(?:play)"
                r"\s+(.+)",
                normalized,
                flags=re.IGNORECASE,
            )

            if match:

                parameters[
                    "media_query"
                ] = match.group(
                    1
                ).strip()

        return parameters

    # ========================================================================
    # DETECT URGENCY
    # ========================================================================

    @staticmethod
    def detect_urgency(
        text: str,
    ) -> int:
        """
        Detect urgency from 0 to 10.
        """

        normalized = (
            text.lower()
        )

        urgency = 0

        urgent_words = {
            "urgent": 4,
            "immediately": 5,
            "emergency": 7,
            "asap": 4,
            "now": 2,
            "quickly": 2,
            "hurry": 3,
            "critical": 6,
            "important": 2,
        }

        for word, score in (
            urgent_words.items()
        ):

            if re.search(
                rf"\b{re.escape(word)}\b",
                normalized,
            ):

                urgency += score

        return min(
            10,
            urgency,
        )

    # ========================================================================
    # DETECT SOURCE
    # ========================================================================

    @staticmethod
    def normalize_source(
        source: str | IntentSource,
    ) -> IntentSource:
        """
        Convert source into IntentSource.
        """

        if isinstance(
            source,
            IntentSource,
        ):

            return source

        normalized = (
            str(source)
            .strip()
            .lower()
        )

        try:

            return IntentSource(
                normalized
            )

        except ValueError:

            return IntentSource.UNKNOWN

    # ========================================================================
    # CUSTOM DETECTOR RESULT
    # ========================================================================

    def _convert_detector_result(
        self,
        result: Any,
        raw_input: str,
        normalized_input: str,
        source: IntentSource,
    ) -> Optional[
        RENIXIntent
    ]:
        """
        Convert custom detector output into RENIXIntent.
        """

        if isinstance(
            result,
            RENIXIntent,
        ):

            return result

        if not isinstance(
            result,
            dict,
        ):

            return None

        name = str(
            result.get(
                "name",
                result.get(
                    "intent",
                    "unknown",
                ),
            )
        )

        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        return RENIXIntent(
            intent_id=(
                self._generate_intent_id()
            ),
            name=name,
            raw_input=raw_input,
            normalized_input=normalized_input,
            confidence=confidence,
            confidence_level=(
                self._confidence_level(
                    confidence
                )
            ),
            source=source,
            category=str(
                result.get(
                    "category",
                    "general",
                )
            ),
            entities=dict(
                result.get(
                    "entities",
                    {},
                )
            ),
            parameters=dict(
                result.get(
                    "parameters",
                    {},
                )
            ),
            keywords=list(
                result.get(
                    "keywords",
                    [],
                )
            ),
            alternatives=list(
                result.get(
                    "alternatives",
                    [],
                )
            ),
            requires_confirmation=bool(
                result.get(
                    "requires_confirmation",
                    False,
                )
            ),
            urgency=int(
                result.get(
                    "urgency",
                    0,
                )
            ),
            destructive=bool(
                result.get(
                    "destructive",
                    False,
                )
            ),
            metadata=dict(
                result.get(
                    "metadata",
                    {},
                )
            ),
        )

    # ========================================================================
    # RULE DETECTION
    # ========================================================================

    def _detect_from_rules(
        self,
        raw_input: str,
        normalized_input: str,
        source: IntentSource,
    ) -> Optional[
        RENIXIntent
    ]:
        """
        Detect intent using local rules.
        """

        tokens = self.tokenize(
            normalized_input
        )

        scored: list[
            tuple[
                float,
                int,
                IntentRule,
            ]
        ] = []

        for rule in (
            self.rules.values()
        ):

            score = self._score_rule(
                rule,
                normalized_input,
                tokens,
            )

            if (
                score
                >= self.minimum_confidence
            ):

                scored.append(
                    (
                        score,
                        rule.priority,
                        rule,
                    )
                )

        if not scored:

            return None

        scored.sort(
            key=lambda item: (
                -item[0],
                -item[1],
            )
        )

        best_score, _, best_rule = (
            scored[0]
        )

        alternatives = []

        for score, _, rule in scored[1:6]:

            alternatives.append(
                {
                    "name": rule.name,
                    "confidence": round(
                        score,
                        4,
                    ),
                    "category": (
                        rule.category
                    ),
                }
            )

        entities = (
            self.extract_entities(
                raw_input
            )
        )

        if best_rule.entities:

            entities.update(
                best_rule.entities
            )

        parameters = (
            self.extract_parameters(
                raw_input,
                best_rule.name,
            )
        )

        urgency = (
            self.detect_urgency(
                normalized_input
            )
        )

        return RENIXIntent(
            intent_id=(
                self._generate_intent_id()
            ),
            name=best_rule.name,
            raw_input=raw_input,
            normalized_input=(
                normalized_input
            ),
            confidence=best_score,
            confidence_level=(
                self._confidence_level(
                    best_score
                )
            ),
            source=source,
            category=best_rule.category,
            entities=entities,
            parameters=parameters,
            keywords=[
                keyword
                for keyword
                in best_rule.keywords
                if keyword in tokens
            ],
            alternatives=alternatives,
            requires_confirmation=(
                best_rule.requires_confirmation
            ),
            urgency=urgency,
            destructive=(
                best_rule.destructive
            ),
            metadata={
                "detector": "rule_engine",
            },
        )

    # ========================================================================
    # AI DETECTION
    # ========================================================================

    async def _detect_with_ai(
        self,
        raw_input: str,
        normalized_input: str,
        source: IntentSource,
    ) -> Optional[
        RENIXIntent
    ]:
        """
        Ask an external AI detector to identify intent.
        """

        if self.ai_detector is None:

            return None

        try:

            result = self.ai_detector(
                raw_input,
                normalized_input,
                source,
            )

            if hasattr(
                result,
                "__await__",
            ):

                result = await result

            return (
                self._convert_detector_result(
                    result,
                    raw_input,
                    normalized_input,
                    source,
                )
            )

        except Exception:

            self.logger.exception(
                "AI intent detection failed."
            )

            return None

    # ========================================================================
    # MAIN DETECT
    # ========================================================================

    async def detect(
        self,
        text: str,
        *,
        source: str | IntentSource = (
            IntentSource.UNKNOWN
        ),
        metadata: Optional[
            dict[str, Any]
        ] = None,
        use_ai: bool = True,
    ) -> RENIXIntent:
        """
        Detect the user's intent.

        Detection order:

            1. Custom detectors
            2. Local rule engine
            3. AI detector
            4. Unknown intent
        """

        self.total_processed += 1

        raw_input = str(
            text or ""
        )

        normalized_input = (
            self.normalize_input(
                raw_input
            )
        )

        normalized_source = (
            self.normalize_source(
                source
            )
        )

        # --------------------------------------------------------------------
        # Empty input
        # --------------------------------------------------------------------

        if not normalized_input:

            self.total_unknown += 1

            intent = RENIXIntent(
                intent_id=(
                    self._generate_intent_id()
                ),
                name="unknown",
                raw_input=raw_input,
                normalized_input="",
                confidence=0.0,
                confidence_level=(
                    IntentConfidence.VERY_LOW
                ),
                source=normalized_source,
                category="unknown",
                metadata={
                    "reason": "empty_input",
                    **(
                        metadata or {}
                    ),
                },
            )

            self._record(
                intent
            )

            return intent

        # --------------------------------------------------------------------
        # Custom detectors
        # --------------------------------------------------------------------

        for detector in (
            self.detectors
        ):

            try:

                result = detector(
                    raw_input,
                    normalized_input,
                    normalized_source,
                )

                if hasattr(
                    result,
                    "__await__",
                ):

                    result = await result

                intent = (
                    self._convert_detector_result(
                        result,
                        raw_input,
                        normalized_input,
                        normalized_source,
                    )
                )

                if (
                    intent is not None
                    and intent.confidence
                    >= self.minimum_confidence
                ):

                    if metadata:

                        intent.metadata.update(
                            metadata
                        )

                    self.total_detected += 1

                    self._record(
                        intent
                    )

                    return intent

            except Exception:

                self.logger.exception(
                    "Custom intent detector failed."
                )

        # --------------------------------------------------------------------
        # Local rules
        # --------------------------------------------------------------------

        rule_intent = (
            self._detect_from_rules(
                raw_input,
                normalized_input,
                normalized_source,
            )
        )

        # --------------------------------------------------------------------
        # AI detector
        # --------------------------------------------------------------------

        ai_intent = None

        if use_ai:

            ai_intent = (
                await self._detect_with_ai(
                    raw_input,
                    normalized_input,
                    normalized_source,
                )
            )

        # --------------------------------------------------------------------
        # Select best result
        # --------------------------------------------------------------------

        candidates = [
            intent
            for intent in (
                rule_intent,
                ai_intent,
            )
            if intent is not None
        ]

        if candidates:

            candidates.sort(
                key=lambda item: (
                    item.confidence
                ),
                reverse=True,
            )

            intent = candidates[0]

            if (
                len(candidates)
                > 1
            ):

                for alternative in (
                    candidates[1:]
                ):

                    intent.alternatives.append(
                        {
                            "name": (
                                alternative.name
                            ),
                            "confidence": (
                                alternative.confidence
                            ),
                            "category": (
                                alternative.category
                            ),
                        }
                    )

            if metadata:

                intent.metadata.update(
                    metadata
                )

            self.total_detected += 1

            self._record(
                intent
            )

            return intent

        # --------------------------------------------------------------------
        # Unknown
        # --------------------------------------------------------------------

        self.total_unknown += 1

        unknown = RENIXIntent(
            intent_id=(
                self._generate_intent_id()
            ),
            name="unknown",
            raw_input=raw_input,
            normalized_input=normalized_input,
            confidence=0.0,
            confidence_level=(
                IntentConfidence.VERY_LOW
            ),
            source=normalized_source,
            category="unknown",
            entities=(
                self.extract_entities(
                    raw_input
                )
            ),
            parameters={},
            keywords=self.tokenize(
                normalized_input
            ),
            urgency=(
                self.detect_urgency(
                    normalized_input
                )
            ),
            metadata={
                "reason": "no_matching_intent",
                **(
                    metadata or {}
                ),
            },
        )

        self._record(
            unknown
        )

        return unknown

    # ========================================================================
    # SYNCHRONOUS DETECT
    # ========================================================================

    def detect_sync(
        self,
        text: str,
        *,
        source: str | IntentSource = (
            IntentSource.UNKNOWN
        ),
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> RENIXIntent:
        """
        Synchronous convenience wrapper.

        Intended for code that is not already running
        inside an asyncio event loop.
        """

        try:

            asyncio.get_running_loop()

        except RuntimeError:

            return asyncio.run(
                self.detect(
                    text,
                    source=source,
                    metadata=metadata,
                )
            )

        raise RuntimeError(
            "detect_sync() cannot be called "
            "from an active asyncio event loop. "
            "Use await detect() instead."
        )

    # ========================================================================
    # RECORD HISTORY
    # ========================================================================

    def _record(
        self,
        intent: RENIXIntent,
    ) -> None:
        """
        Add intent to history.
        """

        self.history.append(
            intent
        )

        if (
            len(self.history)
            > self.max_history
        ):

            self.history = self.history[
                -self.max_history:
            ]

    # ========================================================================
    # HISTORY
    # ========================================================================

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[RENIXIntent]:
        """
        Return intent history.
        """

        if limit is None:

            return list(
                self.history
            )

        limit = max(
            0,
            int(limit),
        )

        return list(
            self.history[
                -limit:
            ]
        )

    # ========================================================================
    # CLEAR HISTORY
    # ========================================================================

    def clear_history(
        self,
    ) -> None:
        """
        Clear intent history.
        """

        self.history.clear()

    # ========================================================================
    # GET RULE
    # ========================================================================

    def get_rule(
        self,
        name: str,
    ) -> Optional[IntentRule]:
        """
        Get an intent rule.
        """

        return self.rules.get(
            self.normalize_input(
                name
            )
        )

    # ========================================================================
    # ENABLE RULE
    # ========================================================================

    def enable_rule(
        self,
        name: str,
    ) -> bool:
        """
        Enable an intent rule.
        """

        rule = self.get_rule(
            name
        )

        if rule is None:

            return False

        rule.enabled = True

        return True

    # ========================================================================
    # DISABLE RULE
    # ========================================================================

    def disable_rule(
        self,
        name: str,
    ) -> bool:
        """
        Disable an intent rule.
        """

        rule = self.get_rule(
            name
        )

        if rule is None:

            return False

        rule.enabled = False

        return True

    # ========================================================================
    # SEARCH RULES
    # ========================================================================

    def search_rules(
        self,
        query: str,
    ) -> list[IntentRule]:
        """
        Search intent rules.
        """

        query = (
            self.normalize_input(
                query
            )
        )

        if not query:

            return []

        results = []

        for rule in (
            self.rules.values()
        ):

            searchable = " ".join(
                [
                    rule.name,
                    rule.category,
                    rule.description
                    if hasattr(
                        rule,
                        "description",
                    )
                    else "",
                    *rule.keywords,
                    *rule.phrases,
                ]
            ).lower()

            if query in searchable:

                results.append(
                    rule
                )

        results.sort(
            key=lambda rule: (
                -rule.priority,
                rule.name,
            )
        )

        return results

    # ========================================================================
    # EXPORT RULES
    # ========================================================================

    def export_rules(
        self,
    ) -> list[dict[str, Any]]:
        """
        Export all intent rules.
        """

        return [
            rule.to_dict()
            for rule in self.rules.values()
        ]

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return intent engine statistics.
        """

        return {
            "registered_rules": len(
                self.rules
            ),
            "custom_detectors": len(
                self.detectors
            ),
            "ai_detector_enabled": (
                self.ai_detector
                is not None
            ),
            "history_size": len(
                self.history
            ),
            "total_processed": (
                self.total_processed
            ),
            "total_detected": (
                self.total_detected
            ),
            "total_unknown": (
                self.total_unknown
            ),
            "minimum_confidence": (
                self.minimum_confidence
            ),
        }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

intent_engine = IntentEngine()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def detect_intent(
    text: str,
    *,
    source: str | IntentSource = (
        IntentSource.UNKNOWN
    ),
    metadata: Optional[
        dict[str, Any]
    ] = None,
    use_ai: bool = True,
) -> RENIXIntent:
    """
    Detect intent using the global RENIX Intent Engine.
    """

    return await intent_engine.detect(
        text,
        source=source,
        metadata=metadata,
        use_ai=use_ai,
    )


def detect_intent_sync(
    text: str,
    *,
    source: str | IntentSource = (
        IntentSource.UNKNOWN
    ),
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> RENIXIntent:
    """
    Synchronous convenience function.
    """

    return intent_engine.detect_sync(
        text,
        source=source,
        metadata=metadata,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "IntentConfidence",
    "IntentSource",
    "RENIXIntent",
    "IntentRule",
    "IntentEngine",
    "intent_engine",
    "detect_intent",
    "detect_intent_sync",
]


