"""
RENIX AI
LLM Streaming Layer

Provides common streaming utilities for RENIX LLM providers.

Responsibilities:

- Stream chunk handling
- Text accumulation
- Callback handling
- Streaming cancellation
- Stream statistics
- Synchronous streaming
- Asynchronous streaming
- Stream-to-response conversion
"""

from __future__ import annotations

import asyncio
import threading
import time

from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
)

from .provider import (
    LLMChunk,
    LLMRequest,
    LLMResponse,
    LLMProvider,
)


# ============================================================================
# TYPES
# ============================================================================

ChunkCallback = Callable[
    [LLMChunk],
    None,
]

AsyncChunkCallback = Callable[
    [LLMChunk],
    Any,
]

CompleteCallback = Callable[
    [LLMResponse],
    None,
]

ErrorCallback = Callable[
    [Exception],
    None,
]


# ============================================================================
# STREAM STATISTICS
# ============================================================================


@dataclass
class StreamStatistics:
    """
    Runtime statistics for an LLM stream.
    """

    started_at: Optional[float] = None

    ended_at: Optional[float] = None

    chunk_count: int = 0

    character_count: int = 0

    token_count: int = 0

    first_chunk_at: Optional[float] = None

    completed: bool = False

    cancelled: bool = False

    failed: bool = False

    error: Optional[str] = None

    def start(self) -> None:
        """
        Mark the stream as started.
        """

        if self.started_at is None:
            self.started_at = time.perf_counter()

    def record_chunk(
        self,
        chunk: LLMChunk,
    ) -> None:
        """
        Record a received chunk.
        """

        now = time.perf_counter()

        if self.started_at is None:
            self.started_at = now

        if self.first_chunk_at is None:
            self.first_chunk_at = now

        self.chunk_count += 1

        self.character_count += len(
            chunk.content or ""
        )

    def finish(
        self,
        *,
        completed: bool = True,
    ) -> None:
        """
        Mark the stream as finished.
        """

        self.ended_at = time.perf_counter()

        self.completed = completed

    def cancel(self) -> None:
        """
        Mark the stream as cancelled.
        """

        self.cancelled = True

        self.ended_at = time.perf_counter()

    def fail(
        self,
        error: Exception,
    ) -> None:
        """
        Mark the stream as failed.
        """

        self.failed = True

        self.error = str(error)

        self.ended_at = time.perf_counter()

    @property
    def duration(self) -> float:
        """
        Return stream duration in seconds.
        """

        if self.started_at is None:
            return 0.0

        end = (
            self.ended_at
            if self.ended_at is not None
            else time.perf_counter()
        )

        return max(
            0.0,
            end - self.started_at,
        )

    @property
    def time_to_first_chunk(self) -> float:
        """
        Return time from stream start until first chunk.
        """

        if (
            self.started_at is None
            or self.first_chunk_at is None
        ):
            return 0.0

        return max(
            0.0,
            self.first_chunk_at
            - self.started_at,
        )

    @property
    def characters_per_second(self) -> float:
        """
        Estimate output characters per second.
        """

        duration = self.duration

        if duration <= 0:
            return 0.0

        return (
            self.character_count
            / duration
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to a dictionary.
        """

        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "chunk_count": self.chunk_count,
            "character_count": self.character_count,
            "token_count": self.token_count,
            "first_chunk_at": self.first_chunk_at,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "failed": self.failed,
            "error": self.error,
            "duration": self.duration,
            "time_to_first_chunk": (
                self.time_to_first_chunk
            ),
            "characters_per_second": (
                self.characters_per_second
            ),
        }


# ============================================================================
# STREAM RESULT
# ============================================================================


@dataclass
class StreamResult:
    """
    Final result produced after consuming a stream.
    """

    response: LLMResponse

    statistics: StreamStatistics

    chunks: List[LLMChunk] = field(
        default_factory=list
    )

    cancelled: bool = False

    error: Optional[Exception] = None


# ============================================================================
# STREAM CONTROLLER
# ============================================================================


class StreamController:
    """
    Controls the lifecycle of an LLM stream.

    Supports:

    - Cancellation
    - Pause state
    - Resume
    - Callback management
    - Statistics
    """

    def __init__(self) -> None:
        self._cancelled = threading.Event()

        self._paused = threading.Event()

        self._paused.clear()

        self.statistics = (
            StreamStatistics()
        )

        self._lock = threading.Lock()

    # ========================================================================
    # STATE
    # ========================================================================

    @property
    def cancelled(self) -> bool:
        """
        Return whether the stream has been cancelled.
        """

        return self._cancelled.is_set()

    @property
    def paused(self) -> bool:
        """
        Return whether the stream is paused.
        """

        return self._paused.is_set()

    # ========================================================================
    # CONTROL
    # ========================================================================

    def cancel(self) -> None:
        """
        Cancel the stream.
        """

        with self._lock:
            self._cancelled.set()

            self.statistics.cancel()

    def pause(self) -> None:
        """
        Pause consumption of stream chunks.
        """

        self._paused.set()

    def resume(self) -> None:
        """
        Resume stream consumption.
        """

        self._paused.clear()

    def wait_if_paused(
        self,
        interval: float = 0.05,
    ) -> None:
        """
        Wait while the stream is paused.
        """

        while self.paused and not self.cancelled:
            time.sleep(interval)


# ============================================================================
# STREAM ACCUMULATOR
# ============================================================================


class StreamAccumulator:
    """
    Accumulates streaming chunks into a complete response.
    """

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:

        self.provider = provider

        self.model = model

        self.chunks: List[
            LLMChunk
        ] = []

        self._parts: List[str] = []

        self._finish_reason: Optional[
            str
        ] = None

        self._response_id: Optional[
            str
        ] = None

        self._metadata: Dict[
            str,
            Any,
        ] = {}

        self._lock = threading.Lock()

    def add(
        self,
        chunk: LLMChunk,
    ) -> None:
        """
        Add a chunk to the accumulator.
        """

        with self._lock:
            self.chunks.append(
                chunk
            )

            if chunk.content:
                self._parts.append(
                    chunk.content
                )

            if chunk.finish_reason:
                self._finish_reason = (
                    chunk.finish_reason
                )

            if chunk.response_id:
                self._response_id = (
                    chunk.response_id
                )

            if chunk.model:
                self.model = chunk.model

            if chunk.provider:
                self.provider = (
                    chunk.provider
                )

            if chunk.metadata:
                self._metadata.update(
                    chunk.metadata
                )

    @property
    def text(self) -> str:
        """
        Return all accumulated text.
        """

        with self._lock:
            return "".join(
                self._parts
            )

    def response(self) -> LLMResponse:
        """
        Build a complete LLMResponse.
        """

        return LLMResponse(
            content=self.text,
            model=self.model,
            provider=self.provider,
            finish_reason=(
                self._finish_reason
            ),
            response_id=(
                self._response_id
            ),
            metadata=dict(
                self._metadata
            ),
            raw_response=list(
                self.chunks
            ),
        )


# ============================================================================
# SYNCHRONOUS STREAM
# ============================================================================


class LLMStream:
    """
    High-level synchronous stream wrapper.

    This object can be iterated directly:

        for chunk in stream:
            print(chunk.content)

    Or consumed completely using ``collect()``.
    """

    def __init__(
        self,
        provider: LLMProvider,
        request: LLMRequest,
        *,
        controller: Optional[
            StreamController
        ] = None,
        on_chunk: Optional[
            ChunkCallback
        ] = None,
        on_complete: Optional[
            CompleteCallback
        ] = None,
        on_error: Optional[
            ErrorCallback
        ] = None,
    ) -> None:

        self.provider = provider

        self.request = request

        self.controller = (
            controller
            or StreamController()
        )

        self.on_chunk = on_chunk

        self.on_complete = (
            on_complete
        )

        self.on_error = on_error

        self.accumulator = (
            StreamAccumulator(
                provider=provider.get_name(),
                model=request.model,
            )
        )

        self._iterator: Optional[
            Iterator[LLMChunk]
        ] = None

        self._started = False

        self._finished = False

    # ========================================================================
    # ITERATION
    # ========================================================================

    def __iter__(
        self,
    ) -> Iterator[LLMChunk]:
        """
        Iterate over streaming chunks.
        """

        if self._iterator is None:
            self._iterator = (
                self._create_iterator()
            )

        return self

    def __next__(
        self,
    ) -> LLMChunk:
        """
        Return the next chunk.
        """

        if self._iterator is None:
            self._iterator = (
                self._create_iterator()
            )

        try:
            chunk = next(
                self._iterator
            )

        except StopIteration:
            self._finish()

            raise

        return chunk

    # ========================================================================
    # INTERNAL STREAM
    # ========================================================================

    def _create_iterator(
        self,
    ) -> Iterator[LLMChunk]:
        """
        Create the underlying provider iterator.
        """

        if self._started:
            raise RuntimeError(
                "Stream can only be started once."
            )

        self._started = True

        self.controller.statistics.start()

        request = self.provider.prepare_request(
            self.request
        )

        request.stream = True

        try:
            for chunk in (
                self.provider.generate_stream(
                    request
                )
            ):
                if self.controller.cancelled:
                    break

                self.controller.wait_if_paused()

                if self.controller.cancelled:
                    break

                self.controller.statistics.record_chunk(
                    chunk
                )

                self.accumulator.add(
                    chunk
                )

                if self.on_chunk:
                    self.on_chunk(
                        chunk
                    )

                yield chunk

                if chunk.is_final:
                    break

        except Exception as exc:
            self.controller.statistics.fail(
                exc
            )

            if self.on_error:
                self.on_error(
                    exc
                )

            raise

    # ========================================================================
    # FINISH
    # ========================================================================

    def _finish(self) -> None:
        """
        Finish the stream exactly once.
        """

        if self._finished:
            return

        self._finished = True

        if self.controller.cancelled:
            return

        self.controller.statistics.finish(
            completed=True
        )

        response = (
            self.accumulator.response()
        )

        if self.on_complete:
            self.on_complete(
                response
            )

    # ========================================================================
    # COLLECTION
    # ========================================================================

    def collect(self) -> StreamResult:
        """
        Consume the entire stream and return its final result.
        """

        error: Optional[
            Exception
        ] = None

        try:
            for _ in self:
                pass

        except Exception as exc:
            error = exc

        return StreamResult(
            response=self.accumulator.response(),
            statistics=self.controller.statistics,
            chunks=list(
                self.accumulator.chunks
            ),
            cancelled=self.controller.cancelled,
            error=error,
        )

    @property
    def text(self) -> str:
        """
        Return currently accumulated text.
        """

        return self.accumulator.text

    @property
    def cancelled(self) -> bool:
        """
        Return whether the stream was cancelled.
        """

        return self.controller.cancelled

    def cancel(self) -> None:
        """
        Cancel the stream.
        """

        self.controller.cancel()

    def pause(self) -> None:
        """
        Pause stream consumption.
        """

        self.controller.pause()

    def resume(self) -> None:
        """
        Resume stream consumption.
        """

        self.controller.resume()


# ============================================================================
# ASYNCHRONOUS STREAM
# ============================================================================


class AsyncLLMStream:
    """
    High-level asynchronous stream wrapper.
    """

    def __init__(
        self,
        provider: LLMProvider,
        request: LLMRequest,
        *,
        controller: Optional[
            StreamController
        ] = None,
        on_chunk: Optional[
            AsyncChunkCallback
        ] = None,
        on_complete: Optional[
            CompleteCallback
        ] = None,
        on_error: Optional[
            ErrorCallback
        ] = None,
    ) -> None:

        self.provider = provider

        self.request = request

        self.controller = (
            controller
            or StreamController()
        )

        self.on_chunk = on_chunk

        self.on_complete = (
            on_complete
        )

        self.on_error = on_error

        self.accumulator = (
            StreamAccumulator(
                provider=provider.get_name(),
                model=request.model,
            )
        )

        self._started = False

        self._finished = False

    # ========================================================================
    # ASYNC ITERATION
    # ========================================================================

    def __aiter__(
        self,
    ) -> AsyncIterator[LLMChunk]:
        """
        Return the asynchronous iterator.
        """

        return self._iterate()

    async def _iterate(
        self,
    ) -> AsyncIterator[LLMChunk]:
        """
        Consume provider chunks asynchronously.
        """

        if self._started:
            raise RuntimeError(
                "Stream can only be started once."
            )

        self._started = True

        self.controller.statistics.start()

        request = self.provider.prepare_request(
            self.request
        )

        request.stream = True

        try:
            async for chunk in (
                self.provider.generate_stream_async(
                    request
                )
            ):
                if self.controller.cancelled:
                    break

                while (
                    self.controller.paused
                    and not self.controller.cancelled
                ):
                    await asyncio.sleep(
                        0.05
                    )

                if self.controller.cancelled:
                    break

                self.controller.statistics.record_chunk(
                    chunk
                )

                self.accumulator.add(
                    chunk
                )

                if self.on_chunk:
                    result = self.on_chunk(
                        chunk
                    )

                    if asyncio.iscoroutine(
                        result
                    ):
                        await result

                yield chunk

                if chunk.is_final:
                    break

        except Exception as exc:
            self.controller.statistics.fail(
                exc
            )

            if self.on_error:
                self.on_error(
                    exc
                )

            raise

        finally:
            await self._finish()

    async def _finish(
        self,
    ) -> None:
        """
        Finish the asynchronous stream.
        """

        if self._finished:
            return

        self._finished = True

        if self.controller.cancelled:
            return

        self.controller.statistics.finish(
            completed=True
        )

        response = (
            self.accumulator.response()
        )

        if self.on_complete:
            result = self.on_complete(
                response
            )

            if asyncio.iscoroutine(
                result
            ):
                await result

    async def collect(
        self,
    ) -> StreamResult:
        """
        Consume the entire asynchronous stream.
        """

        error: Optional[
            Exception
        ] = None

        try:
            async for _ in self:
                pass

        except Exception as exc:
            error = exc

        return StreamResult(
            response=self.accumulator.response(),
            statistics=self.controller.statistics,
            chunks=list(
                self.accumulator.chunks
            ),
            cancelled=self.controller.cancelled,
            error=error,
        )

    @property
    def text(self) -> str:
        """
        Return currently accumulated text.
        """

        return self.accumulator.text

    @property
    def cancelled(self) -> bool:
        """
        Return whether the stream was cancelled.
        """

        return self.controller.cancelled

    def cancel(self) -> None:
        """
        Cancel the stream.
        """

        self.controller.cancel()

    def pause(self) -> None:
        """
        Pause the stream.
        """

        self.controller.pause()

    def resume(self) -> None:
        """
        Resume the stream.
        """

        self.controller.resume()


# ============================================================================
# STREAM FACTORY
# ============================================================================


def create_stream(
    provider: LLMProvider,
    request: LLMRequest,
    *,
    on_chunk: Optional[
        ChunkCallback
    ] = None,
    on_complete: Optional[
        CompleteCallback
    ] = None,
    on_error: Optional[
        ErrorCallback
    ] = None,
) -> LLMStream:
    """
    Create a synchronous LLM stream.
    """

    request.stream = True

    return LLMStream(
        provider,
        request,
        on_chunk=on_chunk,
        on_complete=on_complete,
        on_error=on_error,
    )


def create_async_stream(
    provider: LLMProvider,
    request: LLMRequest,
    *,
    on_chunk: Optional[
        AsyncChunkCallback
    ] = None,
    on_complete: Optional[
        CompleteCallback
    ] = None,
    on_error: Optional[
        ErrorCallback
    ] = None,
) -> AsyncLLMStream:
    """
    Create an asynchronous LLM stream.
    """

    request.stream = True

    return AsyncLLMStream(
        provider,
        request,
        on_chunk=on_chunk,
        on_complete=on_complete,
        on_error=on_error,
    )


# ============================================================================
# STREAM HELPERS
# ============================================================================


def collect_stream(
    chunks: Iterator[LLMChunk],
) -> LLMResponse:
    """
    Consume a synchronous chunk iterator and return
    one complete LLMResponse.
    """

    accumulator = (
        StreamAccumulator()
    )

    for chunk in chunks:
        accumulator.add(
            chunk
        )

    return accumulator.response()


async def collect_stream_async(
    chunks: AsyncIterator[LLMChunk],
) -> LLMResponse:
    """
    Consume an asynchronous chunk iterator and return
    one complete LLMResponse.
    """

    accumulator = (
        StreamAccumulator()
    )

    async for chunk in chunks:
        accumulator.add(
            chunk
        )

    return accumulator.response()


def stream_text(
    chunks: Iterator[LLMChunk],
) -> Iterator[str]:
    """
    Convert LLM chunks into plain text chunks.
    """

    for chunk in chunks:
        if chunk.content:
            yield chunk.content


async def stream_text_async(
    chunks: AsyncIterator[LLMChunk],
) -> AsyncIterator[str]:
    """
    Convert asynchronous LLM chunks into plain text chunks.
    """

    async for chunk in chunks:
        if chunk.content:
            yield chunk.content


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "ChunkCallback",
    "AsyncChunkCallback",
    "CompleteCallback",
    "ErrorCallback",
    "StreamStatistics",
    "StreamResult",
    "StreamController",
    "StreamAccumulator",
    "LLMStream",
    "AsyncLLMStream",
    "create_stream",
    "create_async_stream",
    "collect_stream",
    "collect_stream_async",
    "stream_text",
    "stream_text_async",
]


