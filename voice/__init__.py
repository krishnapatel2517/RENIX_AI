"""
RENIX AI Voice System

The voice package provides all audio and speech capabilities used by RENIX.

Components:
    microphone
        Microphone input and recording management.

    speech_to_text
        Converts spoken audio into text.

    text_to_speech
        Converts RENIX responses into natural speech.

    wake_word
        Detects RENIX wake commands.

    clap_detector
        Detects the configured clap activation sequence.

    voice_authentication
        Handles voice-based authentication.

    speaker_identification
        Identifies known speakers.

    noise_cancellation
        Audio noise reduction and filtering.

    audio_processing
        General-purpose audio processing utilities.

    conversation_mode
        Continuous voice conversation management.

    interruption
        Detects and manages user interruptions while RENIX is speaking.

    language_detection
        Detects the language being spoken.

The package is designed so individual components can operate independently
while also being coordinated by RENIX's core orchestration system.
"""

from __future__ import annotations

from typing import Any


__version__ = "1.0.0"
__author__ = "RENIX AI"
__description__ = (
    "RENIX AI voice interaction and audio processing system."
)


# ---------------------------------------------------------------------------
# Optional component imports
# ---------------------------------------------------------------------------
#
# RENIX should still be able to start if an optional audio dependency is
# unavailable. Therefore imports are guarded rather than making the entire
# package fail during startup.
#

Microphone: Any = None
SpeechToText: Any = None
TextToSpeech: Any = None
WakeWordDetector: Any = None
ClapDetector: Any = None
VoiceAuthenticator: Any = None
SpeakerIdentifier: Any = None
NoiseCanceller: Any = None
AudioProcessor: Any = None
ConversationMode: Any = None
InterruptionManager: Any = None
LanguageDetector: Any = None


def _load_components() -> None:
    """
    Load voice components safely.

    Individual modules may depend on optional packages such as audio
    drivers, speech-recognition libraries, or machine-learning runtimes.
    RENIX should not crash merely because one optional component is
    unavailable.
    """

    global Microphone
    global SpeechToText
    global TextToSpeech
    global WakeWordDetector
    global ClapDetector
    global VoiceAuthenticator
    global SpeakerIdentifier
    global NoiseCanceller
    global AudioProcessor
    global ConversationMode
    global InterruptionManager
    global LanguageDetector

    try:
        from .microphone import Microphone as _Microphone

        Microphone = _Microphone
    except Exception:
        Microphone = None

    try:
        from .speech_to_text import (
            SpeechToText as _SpeechToText,
        )

        SpeechToText = _SpeechToText
    except Exception:
        SpeechToText = None

    try:
        from .text_to_speech import (
            TextToSpeech as _TextToSpeech,
        )

        TextToSpeech = _TextToSpeech
    except Exception:
        TextToSpeech = None

    try:
        from .wake_word import (
            WakeWordDetector as _WakeWordDetector,
        )

        WakeWordDetector = _WakeWordDetector
    except Exception:
        WakeWordDetector = None

    try:
        from .clap_detector import (
            ClapDetector as _ClapDetector,
        )

        ClapDetector = _ClapDetector
    except Exception:
        ClapDetector = None

    try:
        from .voice_authentication import (
            VoiceAuthenticator as _VoiceAuthenticator,
        )

        VoiceAuthenticator = _VoiceAuthenticator
    except Exception:
        VoiceAuthenticator = None

    try:
        from .speaker_identification import (
            SpeakerIdentifier as _SpeakerIdentifier,
        )

        SpeakerIdentifier = _SpeakerIdentifier
    except Exception:
        SpeakerIdentifier = None

    try:
        from .noise_cancellation import (
            NoiseCanceller as _NoiseCanceller,
        )

        NoiseCanceller = _NoiseCanceller
    except Exception:
        NoiseCanceller = None

    try:
        from .audio_processing import (
            AudioProcessor as _AudioProcessor,
        )

        AudioProcessor = _AudioProcessor
    except Exception:
        AudioProcessor = None

    try:
        from .conversation_mode import (
            ConversationMode as _ConversationMode,
        )

        ConversationMode = _ConversationMode
    except Exception:
        ConversationMode = None

    try:
        from .interruption import (
            InterruptionManager as _InterruptionManager,
        )

        InterruptionManager = _InterruptionManager
    except Exception:
        InterruptionManager = None

    try:
        from .language_detection import (
            LanguageDetector as _LanguageDetector,
        )

        LanguageDetector = _LanguageDetector
    except Exception:
        LanguageDetector = None


_load_components()


# ---------------------------------------------------------------------------
# Package information
# ---------------------------------------------------------------------------

def get_version() -> str:
    """Return the RENIX voice package version."""
    return __version__


def get_available_components() -> dict[str, bool]:
    """
    Return the availability status of every voice subsystem.
    """

    return {
        "microphone": Microphone is not None,
        "speech_to_text": SpeechToText is not None,
        "text_to_speech": TextToSpeech is not None,
        "wake_word": WakeWordDetector is not None,
        "clap_detector": ClapDetector is not None,
        "voice_authentication": VoiceAuthenticator is not None,
        "speaker_identification": SpeakerIdentifier is not None,
        "noise_cancellation": NoiseCanceller is not None,
        "audio_processing": AudioProcessor is not None,
        "conversation_mode": ConversationMode is not None,
        "interruption": InterruptionManager is not None,
        "language_detection": LanguageDetector is not None,
    }


def is_voice_system_available() -> bool:
    """
    Determine whether the basic RENIX voice stack is available.

    RENIX requires at minimum microphone input and speech recognition for
    normal voice interaction.
    """

    return (
        Microphone is not None
        and SpeechToText is not None
    )


__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "Microphone",
    "SpeechToText",
    "TextToSpeech",
    "WakeWordDetector",
    "ClapDetector",
    "VoiceAuthenticator",
    "SpeakerIdentifier",
    "NoiseCanceller",
    "AudioProcessor",
    "ConversationMode",
    "InterruptionManager",
    "LanguageDetector",
    "get_version",
    "get_available_components",
    "is_voice_system_available",
]


