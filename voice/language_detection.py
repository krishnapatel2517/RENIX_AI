"""
RENIX AI - Language Detection Engine

Responsibilities:
- Detect the language of user speech/text
- Support multilingual conversations
- Normalize language codes
- Track language confidence
- Detect language from short text
- Detect language from speech-recognition metadata
- Maintain conversation language
- Support automatic language switching
- Provide language information to other RENIX modules

This module is intentionally independent of:
- Speech-to-text
- Text-to-speech
- Microphone capture
- LLM providers
- Conversation orchestration

Those systems can consume this module's results.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger(
    "RENIX.voice.language_detection"
)


# ============================================================================
# ENUMS
# ============================================================================


class LanguageSource(str, Enum):
    """Source used to determine a language."""

    TEXT = "text"
    STT = "stt"
    USER_PREFERENCE = "user_preference"
    SYSTEM = "system"
    CONTEXT = "context"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class DetectionStatus(str, Enum):
    """Result status."""

    DETECTED = "detected"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    EMPTY = "empty"


# ============================================================================
# LANGUAGE DEFINITIONS
# ============================================================================


@dataclass(frozen=True)
class LanguageInfo:
    """Metadata describing a supported language."""

    code: str
    name: str
    native_name: str
    family: str
    script: str
    aliases: tuple[str, ...] = ()
    rtl: bool = False

    def to_dict(self) -> dict[str, Any]:

        return {
            "code": self.code,
            "name": self.name,
            "native_name": self.native_name,
            "family": self.family,
            "script": self.script,
            "aliases": list(self.aliases),
            "rtl": self.rtl,
        }


@dataclass
class LanguageDetectionResult:
    """Result returned by the detection engine."""

    language: Optional[LanguageInfo]

    confidence: float

    status: DetectionStatus

    source: LanguageSource

    text: str = ""

    candidates: list[
        tuple[str, float]
    ] = field(default_factory=list)

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "language": (
                self.language.to_dict()
                if self.language
                else None
            ),
            "confidence": self.confidence,
            "status": self.status.value,
            "source": self.source.value,
            "text": self.text,
            "candidates": [
                {
                    "language": code,
                    "confidence": confidence,
                }
                for code, confidence
                in self.candidates
            ],
            "metadata": self.metadata,
        }


# ============================================================================
# LANGUAGE DATABASE
# ============================================================================


LANGUAGES: dict[str, LanguageInfo] = {

    "en": LanguageInfo(
        code="en",
        name="English",
        native_name="English",
        family="Germanic",
        script="Latin",
        aliases=(
            "english",
            "eng",
        ),
    ),

    "hi": LanguageInfo(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        family="Indo-Aryan",
        script="Devanagari",
        aliases=(
            "hindi",
            "hin",
        ),
    ),

    "gu": LanguageInfo(
        code="gu",
        name="Gujarati",
        native_name="ગુજરાતી",
        family="Indo-Aryan",
        script="Gujarati",
        aliases=(
            "gujarati",
            "guj",
        ),
    ),

    "mr": LanguageInfo(
        code="mr",
        name="Marathi",
        native_name="मराठी",
        family="Indo-Aryan",
        script="Devanagari",
        aliases=(
            "marathi",
            "mar",
        ),
    ),

    "sa": LanguageInfo(
        code="sa",
        name="Sanskrit",
        native_name="संस्कृतम्",
        family="Indo-Aryan",
        script="Devanagari",
        aliases=(
            "sanskrit",
            "san",
        ),
    ),

    "bn": LanguageInfo(
        code="bn",
        name="Bengali",
        native_name="বাংলা",
        family="Indo-Aryan",
        script="Bengali",
        aliases=(
            "bengali",
            "bangla",
        ),
    ),

    "pa": LanguageInfo(
        code="pa",
        name="Punjabi",
        native_name="ਪੰਜਾਬੀ",
        family="Indo-Aryan",
        script="Gurmukhi",
        aliases=(
            "punjabi",
            "panjabi",
        ),
    ),

    "ta": LanguageInfo(
        code="ta",
        name="Tamil",
        native_name="தமிழ்",
        family="Dravidian",
        script="Tamil",
        aliases=("tamil",),
    ),

    "te": LanguageInfo(
        code="te",
        name="Telugu",
        native_name="తెలుగు",
        family="Dravidian",
        script="Telugu",
        aliases=("telugu",),
    ),

    "kn": LanguageInfo(
        code="kn",
        name="Kannada",
        native_name="ಕನ್ನಡ",
        family="Dravidian",
        script="Kannada",
        aliases=("kannada",),
    ),

    "ml": LanguageInfo(
        code="ml",
        name="Malayalam",
        native_name="മലയാളം",
        family="Dravidian",
        script="Malayalam",
        aliases=("malayalam",),
    ),

    "ur": LanguageInfo(
        code="ur",
        name="Urdu",
        native_name="اردو",
        family="Indo-Aryan",
        script="Arabic",
        aliases=("urdu",),
        rtl=True,
    ),

    "ar": LanguageInfo(
        code="ar",
        name="Arabic",
        native_name="العربية",
        family="Semitic",
        script="Arabic",
        aliases=("arabic",),
        rtl=True,
    ),

    "es": LanguageInfo(
        code="es",
        name="Spanish",
        native_name="Español",
        family="Romance",
        script="Latin",
        aliases=("spanish",),
    ),

    "fr": LanguageInfo(
        code="fr",
        name="French",
        native_name="Français",
        family="Romance",
        script="Latin",
        aliases=("french",),
    ),

    "de": LanguageInfo(
        code="de",
        name="German",
        native_name="Deutsch",
        family="Germanic",
        script="Latin",
        aliases=("german",),
    ),

    "it": LanguageInfo(
        code="it",
        name="Italian",
        native_name="Italiano",
        family="Romance",
        script="Latin",
        aliases=("italian",),
    ),

    "pt": LanguageInfo(
        code="pt",
        name="Portuguese",
        native_name="Português",
        family="Romance",
        script="Latin",
        aliases=("portuguese",),
    ),

    "ru": LanguageInfo(
        code="ru",
        name="Russian",
        native_name="Русский",
        family="Slavic",
        script="Cyrillic",
        aliases=("russian",),
    ),

    "ja": LanguageInfo(
        code="ja",
        name="Japanese",
        native_name="日本語",
        family="Japonic",
        script="Japanese",
        aliases=("japanese",),
    ),

    "ko": LanguageInfo(
        code="ko",
        name="Korean",
        native_name="한국어",
        family="Koreanic",
        script="Hangul",
        aliases=("korean",),
    ),

    "zh": LanguageInfo(
        code="zh",
        name="Chinese",
        native_name="中文",
        family="Sino-Tibetan",
        script="Han",
        aliases=(
            "chinese",
            "mandarin",
        ),
    ),
}


# ============================================================================
# SCRIPT RANGES
# ============================================================================


SCRIPT_RANGES = {

    "Devanagari": (
        "\u0900",
        "\u097F",
    ),

    "Gujarati": (
        "\u0A80",
        "\u0AFF",
    ),

    "Bengali": (
        "\u0980",
        "\u09FF",
    ),

    "Gurmukhi": (
        "\u0A00",
        "\u0A7F",
    ),

    "Tamil": (
        "\u0B80",
        "\u0BFF",
    ),

    "Telugu": (
        "\u0C00",
        "\u0C7F",
    ),

    "Kannada": (
        "\u0C80",
        "\u0CFF",
    ),

    "Malayalam": (
        "\u0D00",
        "\u0D7F",
    ),

    "Arabic": (
        "\u0600",
        "\u06FF",
    ),

    "Cyrillic": (
        "\u0400",
        "\u04FF",
    ),

    "Japanese": (
        "\u3040",
        "\u30FF",
    ),

    "Hangul": (
        "\uAC00",
        "\uD7AF",
    ),

    "Han": (
        "\u4E00",
        "\u9FFF",
    ),
}


# ============================================================================
# COMMON WORDS
# ============================================================================


COMMON_WORDS: dict[str, set[str]] = {

    "en": {
        "the",
        "is",
        "are",
        "you",
        "what",
        "why",
        "how",
        "can",
        "please",
        "hello",
        "thanks",
        "thank",
        "this",
        "that",
        "and",
        "for",
        "with",
        "from",
        "want",
        "need",
        "make",
        "give",
        "tell",
    },

    "hi": {
        "है",
        "हैं",
        "का",
        "की",
        "के",
        "और",
        "मैं",
        "आप",
        "तुम",
        "क्या",
        "क्यों",
        "कैसे",
        "मुझे",
        "चाहिए",
        "नहीं",
        "करना",
        "करो",
        "दो",
        "बताओ",
        "नमस्ते",
    },

    "gu": {
        "છે",
        "હું",
        "તમે",
        "શું",
        "કેમ",
        "કેવી",
        "મને",
        "જોઈએ",
        "નથી",
        "અને",
        "આ",
        "એ",
        "કરો",
        "કહો",
        "આપો",
        "નમસ્તે",
    },

    "mr": {
        "आहे",
        "आहेत",
        "मी",
        "तुम्ही",
        "काय",
        "का",
        "कसे",
        "मला",
        "पाहिजे",
        "नाही",
        "आणि",
        "करा",
        "सांगा",
    },

    "sa": {
        "अस्ति",
        "अहम्",
        "त्वम्",
        "किम्",
        "कथम्",
        "मम",
        "भवति",
        "नमः",
    },

    "es": {
        "el",
        "la",
        "los",
        "las",
        "que",
        "es",
        "como",
        "para",
        "hola",
        "gracias",
        "quiero",
        "necesito",
    },

    "fr": {
        "le",
        "la",
        "les",
        "est",
        "que",
        "vous",
        "pour",
        "avec",
        "bonjour",
        "merci",
        "comment",
        "quoi",
    },

    "de": {
        "der",
        "die",
        "das",
        "ist",
        "und",
        "ich",
        "du",
        "sie",
        "was",
        "wie",
        "danke",
        "hallo",
    },

    "it": {
        "il",
        "la",
        "che",
        "è",
        "sono",
        "come",
        "per",
        "ciao",
        "grazie",
        "voglio",
    },

    "pt": {
        "o",
        "a",
        "os",
        "as",
        "que",
        "é",
        "como",
        "para",
        "olá",
        "obrigado",
        "quero",
    },

    "ru": {
        "это",
        "и",
        "я",
        "ты",
        "вы",
        "что",
        "как",
        "для",
        "привет",
        "спасибо",
    },

    "ja": {
        "です",
        "ます",
        "私",
        "あなた",
        "何",
        "どう",
        "こんにちは",
        "ありがとう",
    },

    "ko": {
        "입니다",
        "저",
        "나",
        "너",
        "무엇",
        "어떻게",
        "안녕하세요",
        "감사합니다",
    },

    "zh": {
        "我",
        "你",
        "他",
        "是",
        "什么",
        "怎么",
        "你好",
        "谢谢",
    },

    "ar": {
        "أنا",
        "أنت",
        "هو",
        "هي",
        "ما",
        "كيف",
        "مرحبا",
        "شكرا",
    },
}


# ============================================================================
# LANGUAGE DETECTOR
# ============================================================================


class LanguageDetector:
    """
    Multilingual language detection engine.

    Detection strategy:

    1. Normalize input.
    2. Detect script.
    3. Detect explicit language names.
    4. Compare common words.
    5. Compare Unicode/script patterns.
    6. Use context as a tie breaker.
    7. Return confidence and candidates.
    """

    def __init__(
        self,
        supported_languages: Optional[
            list[str]
        ] = None,
        minimum_confidence: float = 0.45,
    ) -> None:

        self._lock = threading.RLock()

        self.minimum_confidence = (
            minimum_confidence
        )

        if supported_languages is None:

            self.languages = dict(
                LANGUAGES
            )

        else:

            self.languages = {
                code: LANGUAGES[code]
                for code in supported_languages
                if code in LANGUAGES
            }

        self._context_language: Optional[
            str
        ] = None

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:

        if text is None:

            return ""

        text = str(text)

        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        text = text.strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    # ========================================================================
    # SCRIPT DETECTION
    # ========================================================================

    @staticmethod
    def detect_scripts(
        text: str,
    ) -> dict[str, int]:

        counts: dict[str, int] = {}

        for char in text:

            codepoint = ord(char)

            for (
                script,
                bounds,
            ) in SCRIPT_RANGES.items():

                start = ord(
                    bounds[0]
                )

                end = ord(
                    bounds[1]
                )

                if start <= codepoint <= end:

                    counts[script] = (
                        counts.get(
                            script,
                            0,
                        )
                        + 1
                    )

                    break

        return counts

    # ========================================================================
    # LANGUAGE NAME DETECTION
    # ========================================================================

    def detect_explicit_language(
        self,
        text: str,
    ) -> Optional[str]:

        lowered = text.lower()

        for (
            code,
            info,
        ) in self.languages.items():

            if code in lowered:

                # Only accept actual language
                # references where possible.
                if re.search(
                    rf"\b{re.escape(code)}\b",
                    lowered,
                ):

                    return code

            for alias in info.aliases:

                if alias.lower() in lowered:

                    return code

        return None

    # ========================================================================
    # SCRIPT SCORING
    # ========================================================================

    def _script_scores(
        self,
        text: str,
    ) -> dict[str, float]:

        scripts = self.detect_scripts(
            text
        )

        scores: dict[str, float] = {
            code: 0.0
            for code in self.languages
        }

        total_script_chars = sum(
            scripts.values()
        )

        if total_script_chars == 0:

            return scores

        for (
            code,
            info,
        ) in self.languages.items():

            count = scripts.get(
                info.script,
                0,
            )

            if count:

                scores[code] = (
                    count
                    / total_script_chars
                )

        return scores

    # ========================================================================
    # WORD SCORING
    # ========================================================================

    def _word_scores(
        self,
        text: str,
    ) -> dict[str, float]:

        tokens = re.findall(
            r"\w+",
            text.lower(),
            flags=re.UNICODE,
        )

        if not tokens:

            return {
                code: 0.0
                for code in self.languages
            }

        scores: dict[str, float] = {
            code: 0.0
            for code in self.languages
        }

        token_set = set(
            tokens
        )

        for (
            code,
            words,
        ) in COMMON_WORDS.items():

            if code not in self.languages:

                continue

            matches = (
                token_set
                & words
            )

            if matches:

                scores[code] = min(
                    1.0,
                    len(matches)
                    / max(
                        1,
                        min(
                            len(tokens),
                            8,
                        ),
                    ),
                )

        return scores

    # ========================================================================
    # LATIN LANGUAGE SCORING
    # ========================================================================

    def _latin_scores(
        self,
        text: str,
    ) -> dict[str, float]:

        scores: dict[str, float] = {
            code: 0.0
            for code in self.languages
        }

        lowered = text.lower()

        keyword_map = {

            "en": {
                "the",
                "this",
                "that",
                "what",
                "why",
                "how",
                "please",
                "can",
                "could",
                "would",
                "will",
            },

            "es": {
                "que",
                "cómo",
                "como",
                "para",
                "quiero",
                "hola",
                "gracias",
                "por",
            },

            "fr": {
                "que",
                "pour",
                "avec",
                "bonjour",
                "merci",
                "vous",
                "comment",
            },

            "de": {
                "der",
                "die",
                "das",
                "und",
                "ich",
                "nicht",
                "wie",
                "danke",
            },

            "it": {
                "che",
                "come",
                "per",
                "ciao",
                "grazie",
                "voglio",
            },

            "pt": {
                "que",
                "como",
                "para",
                "olá",
                "obrigado",
                "quero",
            },
        }

        tokens = set(
            re.findall(
                r"\b[a-zA-ZÀ-ÿ]+\b",
                lowered,
            )
        )

        for code, keywords in (
            keyword_map.items()
        ):

            if code not in self.languages:

                continue

            matches = (
                tokens
                & keywords
            )

            if matches:

                scores[code] = min(
                    1.0,
                    len(matches)
                    / 4.0,
                )

        return scores

    # ========================================================================
    # CONTEXT
    # ========================================================================

    def set_context_language(
        self,
        language: Optional[str],
    ) -> None:

        if language is None:

            with self._lock:

                self._context_language = None

            return

        normalized = (
            self.normalize_code(
                language
            )
        )

        if normalized not in self.languages:

            raise ValueError(
                f"Unsupported language: {language}"
            )

        with self._lock:

            self._context_language = (
                normalized
            )

    def get_context_language(
        self,
    ) -> Optional[LanguageInfo]:

        with self._lock:

            code = (
                self._context_language
            )

        if code is None:

            return None

        return self.languages.get(
            code
        )

    # ========================================================================
    # DETECTION
    # ========================================================================

    def detect(
        self,
        text: str,
        *,
        source: LanguageSource = (
            LanguageSource.TEXT
        ),
        context_language: Optional[
            str
        ] = None,
    ) -> LanguageDetectionResult:

        normalized = self.normalize_text(
            text
        )

        if not normalized:

            return LanguageDetectionResult(
                language=None,
                confidence=0.0,
                status=DetectionStatus.EMPTY,
                source=source,
                text="",
            )

        explicit = (
            self.detect_explicit_language(
                normalized
            )
        )

        if explicit:

            language = self.languages.get(
                explicit
            )

            result = (
                LanguageDetectionResult(
                    language=language,
                    confidence=0.99,
                    status=DetectionStatus.DETECTED,
                    source=source,
                    text=normalized,
                    candidates=[
                        (
                            explicit,
                            0.99,
                        )
                    ],
                    metadata={
                        "method": "explicit",
                    },
                )
            )

            self._update_context(
                explicit
            )

            return result

        script_scores = (
            self._script_scores(
                normalized
            )
        )

        word_scores = (
            self._word_scores(
                normalized
            )
        )

        latin_scores = (
            self._latin_scores(
                normalized
            )
        )

        scores: dict[str, float] = {}

        for code in self.languages:

            script_score = (
                script_scores.get(
                    code,
                    0.0,
                )
            )

            word_score = (
                word_scores.get(
                    code,
                    0.0,
                )
            )

            latin_score = (
                latin_scores.get(
                    code,
                    0.0,
                )
            )

            score = (
                script_score * 0.55
                + word_score * 0.30
                + latin_score * 0.15
            )

            scores[code] = score

        # Context is a weak tie-breaker.
        context_code = (
            context_language
            or self._context_language
        )

        if context_code:

            context_code = (
                self.normalize_code(
                    context_code
                )
            )

            if (
                context_code
                in scores
            ):

                scores[
                    context_code
                ] += 0.05

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        if not ranked:

            return LanguageDetectionResult(
                language=None,
                confidence=0.0,
                status=DetectionStatus.UNKNOWN,
                source=source,
                text=normalized,
            )

        best_code, best_score = (
            ranked[0]
        )

        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0.0
        )

        # Avoid pretending confidence is
        # stronger than the evidence.
        confidence = max(
            0.0,
            min(
                1.0,
                best_score,
            ),
        )

        uncertain = (
            confidence
            < self.minimum_confidence
            or (
                confidence
                - second_score
                < 0.08
                and confidence < 0.65
            )
        )

        language = self.languages.get(
            best_code
        )

        status = (
            DetectionStatus.UNCERTAIN
            if uncertain
            else DetectionStatus.DETECTED
        )

        if uncertain:

            language_for_result = (
                language
                if confidence
                >= self.minimum_confidence
                else None
            )

        else:

            language_for_result = language

        result = (
            LanguageDetectionResult(
                language=language_for_result,
                confidence=confidence,
                status=status,
                source=source,
                text=normalized,
                candidates=ranked[:5],
                metadata={
                    "method": "script_and_words",
                    "scripts": self.detect_scripts(
                        normalized
                    ),
                },
            )
        )

        if (
            language_for_result is not None
            and confidence
            >= self.minimum_confidence
        ):

            self._update_context(
                language_for_result.code
            )

        return result

    # ========================================================================
    # STT RESULT DETECTION
    # ========================================================================

    def detect_from_stt(
        self,
        text: str,
        *,
        language_hint: Optional[
            str
        ] = None,
        confidence: Optional[
            float
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> LanguageDetectionResult:

        if language_hint:

            normalized_hint = (
                self.normalize_code(
                    language_hint
                )
            )

            if (
                normalized_hint
                in self.languages
            ):

                hint_confidence = (
                    confidence
                    if confidence is not None
                    else 0.90
                )

                result = (
                    LanguageDetectionResult(
                        language=self.languages[
                            normalized_hint
                        ],
                        confidence=min(
                            1.0,
                            max(
                                0.0,
                                hint_confidence,
                            ),
                        ),
                        status=DetectionStatus.DETECTED,
                        source=LanguageSource.STT,
                        text=self.normalize_text(
                            text
                        ),
                        candidates=[
                            (
                                normalized_hint,
                                hint_confidence,
                            )
                        ],
                        metadata={
                            **dict(
                                metadata
                                or {}
                            ),
                            "method": (
                                "stt_hint"
                            ),
                        },
                    )
                )

                self._update_context(
                    normalized_hint
                )

                return result

        result = self.detect(
            text,
            source=LanguageSource.STT,
        )

        if metadata:

            result.metadata.update(
                metadata
            )

        return result

    # ========================================================================
    # LANGUAGE CODE NORMALIZATION
    # ========================================================================

    @staticmethod
    def normalize_code(
        code: str,
    ) -> str:

        if not code:

            return ""

        value = str(
            code
        ).strip().lower()

        aliases = {

            "eng": "en",
            "english": "en",

            "hin": "hi",
            "hindi": "hi",

            "guj": "gu",
            "gujarati": "gu",

            "mar": "mr",
            "marathi": "mr",

            "san": "sa",
            "sanskrit": "sa",

            "ben": "bn",
            "bengali": "bn",
            "bangla": "bn",

            "pan": "pa",
            "punjabi": "pa",
            "panjabi": "pa",

            "tam": "ta",
            "tamil": "ta",

            "tel": "te",
            "telugu": "te",

            "kan": "kn",
            "kannada": "kn",

            "mal": "ml",
            "malayalam": "ml",

            "urd": "ur",
            "urdu": "ur",

            "ara": "ar",
            "arabic": "ar",

            "spa": "es",
            "spanish": "es",

            "fra": "fr",
            "fre": "fr",
            "french": "fr",

            "deu": "de",
            "ger": "de",
            "german": "de",

            "ita": "it",
            "italian": "it",

            "por": "pt",
            "portuguese": "pt",

            "rus": "ru",
            "russian": "ru",

            "jpn": "ja",
            "japanese": "ja",

            "kor": "ko",
            "korean": "ko",

            "zho": "zh",
            "chi": "zh",
            "chinese": "zh",
            "mandarin": "zh",
        }

        # Handle forms such as en-US,
        # hi-IN, etc.
        base = value.split(
            "-",
            1,
        )[0]

        base = base.split(
            "_",
            1,
        )[0]

        return aliases.get(
            value,
            aliases.get(
                base,
                base,
            ),
        )

    # ========================================================================
    # LANGUAGE LOOKUP
    # ========================================================================

    def get_language(
        self,
        code: str,
    ) -> Optional[LanguageInfo]:

        normalized = (
            self.normalize_code(
                code
            )
        )

        return self.languages.get(
            normalized
        )

    def list_languages(
        self,
    ) -> list[LanguageInfo]:

        return list(
            self.languages.values()
        )

    # ========================================================================
    # CONTEXT UPDATE
    # ========================================================================

    def _update_context(
        self,
        code: str,
    ) -> None:

        with self._lock:

            self._context_language = code

    # ========================================================================
    # RESET
    # ========================================================================

    def reset_context(
        self,
    ) -> None:

        with self._lock:

            self._context_language = None


# ============================================================================
# LANGUAGE MANAGER
# ============================================================================


class LanguageManager:
    """
    High-level multilingual language manager.

    This is the interface other RENIX systems should
    normally use instead of directly manipulating
    LanguageDetector.
    """

    def __init__(
        self,
        detector: Optional[
            LanguageDetector
        ] = None,
    ) -> None:

        self.detector = (
            detector
            or LanguageDetector()
        )

        self._lock = threading.RLock()

        self._current_language: Optional[
            str
        ] = None

        self._preferred_language: Optional[
            str
        ] = None

        self._auto_detection = True

        self._callbacks: list[
            Callable[
                [LanguageDetectionResult],
                Any,
            ]
        ] = []

    # ========================================================================
    # PREFERENCES
    # ========================================================================

    def set_preferred_language(
        self,
        language: str,
    ) -> None:

        normalized = (
            self.detector.normalize_code(
                language
            )
        )

        if (
            normalized
            not in self.detector.languages
        ):

            raise ValueError(
                f"Unsupported language: {language}"
            )

        with self._lock:

            self._preferred_language = (
                normalized
            )

    def get_preferred_language(
        self,
    ) -> Optional[
        LanguageInfo
    ]:

        with self._lock:

            code = (
                self._preferred_language
            )

        if code is None:

            return None

        return self.detector.get_language(
            code
        )

    # ========================================================================
    # CURRENT LANGUAGE
    # ========================================================================

    def get_current_language(
        self,
    ) -> Optional[
        LanguageInfo
    ]:

        with self._lock:

            code = (
                self._current_language
            )

        if code is None:

            return None

        return self.detector.get_language(
            code
        )

    def set_current_language(
        self,
        language: str,
    ) -> None:

        normalized = (
            self.detector.normalize_code(
                language
            )
        )

        if (
            normalized
            not in self.detector.languages
        ):

            raise ValueError(
                f"Unsupported language: {language}"
            )

        with self._lock:

            self._current_language = (
                normalized
            )

        self.detector.set_context_language(
            normalized
        )

    # ========================================================================
    # AUTO DETECTION
    # ========================================================================

    def enable_auto_detection(
        self,
    ) -> None:

        with self._lock:

            self._auto_detection = True

    def disable_auto_detection(
        self,
    ) -> None:

        with self._lock:

            self._auto_detection = False

    @property
    def auto_detection_enabled(
        self,
    ) -> bool:

        with self._lock:

            return self._auto_detection

    # ========================================================================
    # DETECTION
    # ========================================================================

    def process_text(
        self,
        text: str,
        *,
        source: LanguageSource = (
            LanguageSource.TEXT
        ),
    ) -> LanguageDetectionResult:

        result = self.detector.detect(
            text,
            source=source,
            context_language=(
                self._current_language
            ),
        )

        if (
            self.auto_detection_enabled
            and result.language is not None
            and result.confidence
            >= self.detector.minimum_confidence
        ):

            changed = False

            with self._lock:

                if (
                    self._current_language
                    != result.language.code
                ):

                    changed = True

                self._current_language = (
                    result.language.code
                )

            if changed:

                self._notify_language_change(
                    result
                )

        return result

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def register_language_callback(
        self,
        callback: Callable[
            [LanguageDetectionResult],
            Any,
        ],
    ) -> None:

        if not callable(callback):

            raise TypeError(
                "callback must be callable."
            )

        with self._lock:

            self._callbacks.append(
                callback
            )

    def unregister_language_callback(
        self,
        callback: Callable[
            [LanguageDetectionResult],
            Any,
        ],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    def _notify_language_change(
        self,
        result: LanguageDetectionResult,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks
            )

        for callback in callbacks:

            try:

                callback(result)

            except Exception:

                logger.exception(
                    "Language change callback failed."
                )

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self,
    ) -> dict[str, Any]:

        current = (
            self.get_current_language()
        )

        preferred = (
            self.get_preferred_language()
        )

        return {
            "auto_detection": (
                self.auto_detection_enabled
            ),
            "current_language": (
                current.to_dict()
                if current
                else None
            ),
            "preferred_language": (
                preferred.to_dict()
                if preferred
                else None
            ),
            "supported_languages": [
                language.code
                for language
                in self.detector.list_languages()
            ],
        }

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(
        self,
    ) -> None:

        with self._lock:

            self._current_language = None

        self.detector.reset_context()


# ============================================================================
# DEFAULT GLOBAL INSTANCES
# ============================================================================


_detector: Optional[
    LanguageDetector
] = None

_manager: Optional[
    LanguageManager
] = None

_global_lock = threading.RLock()


def get_language_detector() -> LanguageDetector:

    global _detector

    with _global_lock:

        if _detector is None:

            _detector = LanguageDetector()

        return _detector


def get_language_manager() -> LanguageManager:

    global _manager

    with _global_lock:

        if _manager is None:

            _manager = LanguageManager(
                get_language_detector()
            )

        return _manager


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def detect_language(
    text: str,
) -> LanguageDetectionResult:

    return (
        get_language_detector()
        .detect(text)
    )


def detect_language_code(
    text: str,
) -> Optional[str]:

    result = detect_language(
        text
    )

    if result.language is None:

        return None

    return result.language.code


def get_language(
    code: str,
) -> Optional[LanguageInfo]:

    return (
        get_language_detector()
        .get_language(code)
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "LanguageSource",
    "DetectionStatus",
    "LanguageInfo",
    "LanguageDetectionResult",
    "LanguageDetector",
    "LanguageManager",
    "LANGUAGES",
    "get_language_detector",
    "get_language_manager",
    "detect_language",
    "detect_language_code",
    "get_language",
]


