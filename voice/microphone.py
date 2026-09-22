"""
RENIX AI - Microphone Manager

Handles microphone discovery, configuration, recording, streaming,
device selection, and audio capture.

Designed to work as the input layer for:
    microphone -> audio processing -> speech-to-text

The implementation uses sounddevice when available, while keeping
the module importable when optional audio dependencies are missing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger("RENIX.voice.microphone")


@dataclass
class MicrophoneConfig:
    """Configuration for microphone capture."""

    device: Optional[Any] = None
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    block_size: int = 1024

    # Maximum recording duration.
    max_duration: float = 60.0

    # Input volume multiplier applied by RENIX.
    gain: float = 1.0

    # Whether input clipping should be prevented.
    prevent_clipping: bool = True

    # Queue size for streaming audio.
    queue_size: int = 32


class MicrophoneError(RuntimeError):
    """Base microphone error."""


class MicrophoneUnavailableError(
    MicrophoneError
):
    """Raised when no usable microphone is available."""


class MicrophonePermissionError(
    MicrophoneError
):
    """Raised when microphone access is denied."""


class MicrophoneManager:
    """
    Main RENIX microphone controller.

    Features:
        - Microphone discovery
        - Default-device detection
        - Device switching
        - Audio recording
        - Streaming audio
        - Start/stop controls
        - Callback support
        - RMS level measurement
        - Gain control
        - Thread-safe operation
    """

    def __init__(
        self,
        config: Optional[
            MicrophoneConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or MicrophoneConfig()
        )

        self._sd = None
        self._np = None

        self._stream = None

        self._recording = False
        self._streaming = False

        self._stop_event = threading.Event()

        self._audio_queue: queue.Queue = (
            queue.Queue(
                maxsize=max(
                    1,
                    self.config.queue_size,
                )
            )
        )

        self._lock = threading.RLock()

        self._last_level = 0.0

        self._recording_started_at = None

        self._callback: Optional[
            Callable[[Any], None]
        ] = None

        self._load_dependencies()

    # ------------------------------------------------------------------
    # Dependency loading
    # ------------------------------------------------------------------

    def _load_dependencies(self) -> None:
        """Load optional audio dependencies."""

        try:

            import sounddevice as sd

            self._sd = sd

        except ImportError:

            logger.warning(
                "sounddevice is not installed."
            )

        try:

            import numpy as np

            self._np = np

        except ImportError:

            logger.warning(
                "numpy is not installed."
            )

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Return whether microphone support is available."""

        return (
            self._sd is not None
            and self._np is not None
        )

    def is_available(self) -> bool:
        return self.available

    def require_available(self) -> None:

        if not self.available:

            raise MicrophoneUnavailableError(
                "Microphone support requires "
                "sounddevice and numpy."
            )

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def list_devices(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return all available audio devices.
        """

        self.require_available()

        try:

            devices = (
                self._sd.query_devices()
            )

        except Exception as exc:

            raise MicrophoneError(
                "Unable to query audio devices."
            ) from exc

        result = []

        for index, device in enumerate(
            devices
        ):

            if not isinstance(
                device,
                dict,
            ):
                continue

            result.append(
                {
                    "index": index,
                    "name": device.get(
                        "name",
                        f"Device {index}",
                    ),
                    "max_input_channels": device.get(
                        "max_input_channels",
                        0,
                    ),
                    "max_output_channels": device.get(
                        "max_output_channels",
                        0,
                    ),
                    "default_samplerate": device.get(
                        "default_samplerate"
                    ),
                    "hostapi": device.get(
                        "hostapi"
                    ),
                }
            )

        return result

    def list_input_devices(
        self,
    ) -> list[dict[str, Any]]:
        """Return only devices capable of microphone input."""

        return [
            device
            for device in self.list_devices()
            if device[
                "max_input_channels"
            ]
            > 0
        ]

    def get_default_device(
        self,
    ) -> Optional[dict[str, Any]]:
        """Return the default input device."""

        self.require_available()

        try:

            default = (
                self._sd.default.device
            )

            if isinstance(
                default,
                (list, tuple),
            ):

                input_index = default[0]

            else:

                input_index = default

            devices = self.list_devices()

            for device in devices:

                if (
                    device["index"]
                    == input_index
                ):
                    return device

        except Exception:

            logger.exception(
                "Failed to determine default microphone."
            )

        return None

    def set_device(
        self,
        device: Any,
    ) -> bool:
        """
        Set the active microphone device.

        `device` can be:
            - device index
            - device name
        """

        self.require_available()

        if self.is_recording():
            raise MicrophoneError(
                "Cannot change microphone while recording."
            )

        devices = self.list_input_devices()

        selected = None

        if isinstance(
            device,
            int,
        ):

            for item in devices:

                if item["index"] == device:
                    selected = item
                    break

        else:

            target = str(
                device
            ).strip().lower()

            for item in devices:

                if (
                    item["name"]
                    .strip()
                    .lower()
                    == target
                ):
                    selected = item
                    break

        if selected is None:

            raise MicrophoneError(
                f"Input device not found: {device}"
            )

        self.config.device = selected[
            "index"
        ]

        return True

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        *,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        dtype: Optional[str] = None,
        block_size: Optional[int] = None,
        gain: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> MicrophoneConfig:
        """Update microphone configuration."""

        if sample_rate is not None:

            if sample_rate <= 0:
                raise ValueError(
                    "sample_rate must be positive."
                )

            self.config.sample_rate = int(
                sample_rate
            )

        if channels is not None:

            if channels <= 0:
                raise ValueError(
                    "channels must be positive."
                )

            self.config.channels = int(
                channels
            )

        if dtype is not None:

            self.config.dtype = str(
                dtype
            )

        if block_size is not None:

            if block_size <= 0:
                raise ValueError(
                    "block_size must be positive."
                )

            self.config.block_size = int(
                block_size
            )

        if gain is not None:

            if gain < 0:
                raise ValueError(
                    "gain cannot be negative."
                )

            self.config.gain = float(
                gain
            )

        if max_duration is not None:

            if max_duration <= 0:
                raise ValueError(
                    "max_duration must be positive."
                )

            self.config.max_duration = float(
                max_duration
            )

        return self.config

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        duration: float,
        *,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> Any:
        """
        Record microphone audio for a fixed duration.

        Returns a NumPy array containing the captured audio.
        """

        self.require_available()

        duration = float(duration)

        if duration <= 0:
            raise ValueError(
                "Recording duration must be positive."
            )

        if duration > self.config.max_duration:

            raise ValueError(
                "Recording duration exceeds "
                f"maximum of "
                f"{self.config.max_duration} seconds."
            )

        with self._lock:

            if self._recording:

                raise MicrophoneError(
                    "Microphone is already recording."
                )

            self._recording = True
            self._recording_started_at = (
                time.monotonic()
            )

        rate = (
            int(sample_rate)
            if sample_rate is not None
            else self.config.sample_rate
        )

        channel_count = (
            int(channels)
            if channels is not None
            else self.config.channels
        )

        frames = int(
            duration * rate
        )

        try:

            audio = self._sd.rec(
                frames,
                samplerate=rate,
                channels=channel_count,
                dtype=self.config.dtype,
                device=self.config.device,
            )

            self._sd.wait()

            audio = self._apply_gain(
                audio
            )

            self._last_level = (
                self.calculate_rms(
                    audio
                )
            )

            return audio

        except Exception as exc:

            message = str(
                exc
            ).lower()

            if (
                "permission"
                in message
                or "access"
                in message
            ):

                raise MicrophonePermissionError(
                    "Microphone access was denied."
                ) from exc

            raise MicrophoneError(
                f"Microphone recording failed: {exc}"
            ) from exc

        finally:

            with self._lock:

                self._recording = False
                self._recording_started_at = None

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def start_stream(
        self,
        callback: Optional[
            Callable[[Any], None]
        ] = None,
    ) -> None:
        """
        Start continuous microphone streaming.

        Each audio block is placed in the internal queue and optionally
        passed to `callback`.
        """

        self.require_available()

        with self._lock:

            if self._streaming:

                return

            self._stop_event.clear()

            self._callback = callback

            try:

                self._stream = (
                    self._sd.InputStream(
                        samplerate=(
                            self.config.sample_rate
                        ),
                        channels=(
                            self.config.channels
                        ),
                        dtype=self.config.dtype,
                        blocksize=(
                            self.config.block_size
                        ),
                        device=self.config.device,
                        callback=(
                            self._stream_callback
                        ),
                    )
                )

                self._stream.start()

                self._streaming = True

            except Exception as exc:

                self._stream = None

                message = str(
                    exc
                ).lower()

                if (
                    "permission"
                    in message
                    or "access"
                    in message
                ):

                    raise MicrophonePermissionError(
                        "Microphone access was denied."
                    ) from exc

                raise MicrophoneError(
                    f"Unable to start microphone stream: {exc}"
                ) from exc

    def stop_stream(self) -> None:
        """Stop continuous microphone streaming."""

        with self._lock:

            self._stop_event.set()

            stream = self._stream

            self._stream = None

            self._streaming = False

        if stream is not None:

            try:
                stream.stop()
            except Exception:
                logger.exception(
                    "Error stopping microphone stream."
                )

            try:
                stream.close()
            except Exception:
                logger.exception(
                    "Error closing microphone stream."
                )

    def is_streaming(self) -> bool:
        return self._streaming

    # ------------------------------------------------------------------
    # Stream callback
    # ------------------------------------------------------------------

    def _stream_callback(
        self,
        indata: Any,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:

        if status:

            logger.debug(
                "Microphone stream status: %s",
                status,
            )

        try:

            audio = indata.copy()

            audio = self._apply_gain(
                audio
            )

            self._last_level = (
                self.calculate_rms(
                    audio
                )
            )

            try:

                self._audio_queue.put_nowait(
                    audio
                )

            except queue.Full:

                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    self._audio_queue.put_nowait(
                        audio
                    )
                except queue.Full:
                    pass

            callback = self._callback

            if callback is not None:

                try:
                    callback(audio)
                except Exception:
                    logger.exception(
                        "Microphone callback failed."
                    )

        except Exception:

            logger.exception(
                "Microphone stream callback failed."
            )

    # ------------------------------------------------------------------
    # Audio blocks
    # ------------------------------------------------------------------

    def get_audio(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Retrieve the next audio block from the stream.
        """

        try:

            return self._audio_queue.get(
                timeout=timeout
            )

        except queue.Empty:

            return None

    def audio_generator(
        self,
        *,
        timeout: float = 1.0,
    ) -> Iterator[Any]:
        """
        Yield microphone audio blocks continuously.

        Stops when microphone streaming is stopped.
        """

        while self.is_streaming():

            audio = self.get_audio(
                timeout=timeout
            )

            if audio is not None:
                yield audio

    # ------------------------------------------------------------------
    # Levels
    # ------------------------------------------------------------------

    def get_input_level(self) -> float:
        """
        Return the latest normalized RMS input level.

        Typical range:
            0.0 = silence
            1.0 = very loud / near clipping
        """

        return float(
            max(
                0.0,
                min(
                    1.0,
                    self._last_level,
                ),
            )
        )

    @staticmethod
    def calculate_rms(
        audio: Any,
    ) -> float:
        """Calculate RMS amplitude."""

        if audio is None:
            return 0.0

        try:

            import numpy as np

            values = np.asarray(
                audio,
                dtype=np.float32,
            )

            if values.size == 0:
                return 0.0

            rms = float(
                np.sqrt(
                    np.mean(
                        np.square(values)
                    )
                )
            )

            if not np.isfinite(rms):
                return 0.0

            return max(
                0.0,
                min(
                    1.0,
                    rms,
                ),
            )

        except Exception:

            return 0.0

    # ------------------------------------------------------------------
    # Gain
    # ------------------------------------------------------------------

    def _apply_gain(
        self,
        audio: Any,
    ) -> Any:

        gain = self.config.gain

        if gain == 1.0:
            return audio

        try:

            values = (
                audio * gain
            )

            if (
                self.config.prevent_clipping
            ):

                values = self._np.clip(
                    values,
                    -1.0,
                    1.0,
                )

            return values

        except Exception:

            logger.exception(
                "Failed to apply microphone gain."
            )

            return audio

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_recording(self) -> bool:
        return self._recording

    def get_recording_duration(
        self,
    ) -> float:

        started = (
            self._recording_started_at
        )

        if started is None:
            return 0.0

        return max(
            0.0,
            time.monotonic()
            - started,
        )

    def get_status(self) -> dict[str, Any]:
        """Return complete microphone status."""

        device = None

        if self.available:

            try:
                device = (
                    self.get_default_device()
                    if self.config.device
                    is None
                    else next(
                        (
                            item
                            for item
                            in self.list_input_devices()
                            if item["index"]
                            == self.config.device
                        ),
                        None,
                    )
                )
            except Exception:
                device = None

        return {
            "available": self.available,
            "recording": self.is_recording(),
            "streaming": self.is_streaming(),
            "device": device,
            "device_id": self.config.device,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "dtype": self.config.dtype,
            "block_size": self.config.block_size,
            "gain": self.config.gain,
            "input_level": self.get_input_level(),
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Safely shut down microphone resources."""

        try:
            self.stop_stream()
        finally:

            with self._lock:

                self._recording = False
                self._callback = None

            self._clear_queue()

    close = shutdown

    def _clear_queue(self) -> None:

        while True:

            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break


# ---------------------------------------------------------------------------
# Backward-compatible/simple name
# ---------------------------------------------------------------------------

Microphone = MicrophoneManager


__all__ = [
    "MicrophoneConfig",
    "MicrophoneError",
    "MicrophoneUnavailableError",
    "MicrophonePermissionError",
    "MicrophoneManager",
    "Microphone",
]


