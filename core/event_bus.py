"""
RENIX AI
Core Event Bus

Central asynchronous event system for RENIX.

Responsibilities:
- Register event listeners
- Remove event listeners
- Publish events
- Emit events
- Support synchronous and asynchronous handlers
- Support one-time listeners
- Event history
- Event metadata
- Wildcard event listeners
- Safe handler execution
- Event statistics
- Shutdown and cleanup
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


logger = logging.getLogger(
    "RENIX.EventBus"
)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

EventHandler = Callable[
    ["Event"],
    Any,
]


# ============================================================================
# EVENT
# ============================================================================

@dataclass
class Event:
    """
    Represents an event inside RENIX.
    """

    name: str

    data: Any = None

    event_id: str = field(
        default_factory=lambda:
        f"event_{uuid.uuid4().hex}"
    )

    source: Optional[str] = None

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    cancelled: bool = False

    propagation_stopped: bool = False

    def stop_propagation(
        self,
    ) -> None:
        """
        Stop further event propagation.
        """

        self.propagation_stopped = True

    def cancel(
        self,
    ) -> None:
        """
        Mark the event as cancelled.
        """

        self.cancelled = True

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert the event into a dictionary.
        """

        return {
            "event_id": self.event_id,
            "name": self.name,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": dict(
                self.metadata
            ),
            "cancelled": self.cancelled,
            "propagation_stopped": (
                self.propagation_stopped
            ),
        }


RENIXEvent = Event


# ============================================================================
# EVENT LISTENER
# ============================================================================

@dataclass
class EventListener:
    """
    Represents a registered event listener.
    """

    event_name: str

    handler: EventHandler

    listener_id: str = field(
        default_factory=lambda:
        f"listener_{uuid.uuid4().hex}"
    )

    priority: int = 0

    once: bool = False

    enabled: bool = True

    created_at: float = field(
        default_factory=time.time
    )

    call_count: int = 0

    error_count: int = 0

    last_called_at: Optional[float] = None

    last_error: Optional[str] = None

    def matches(
        self,
        event_name: str,
    ) -> bool:
        """
        Check whether this listener matches an event.
        """

        if self.event_name == "*":
            return True

        if self.event_name == event_name:
            return True

        # --------------------------------------------------------------------
        # Wildcard suffix
        #
        # Example:
        #   "computer.*"
        # matches:
        #   "computer.open"
        #   "computer.close"
        # --------------------------------------------------------------------

        if self.event_name.endswith(
            ".*"
        ):

            prefix = self.event_name[
                :-2
            ]

            return (
                event_name == prefix
                or event_name.startswith(
                    prefix + "."
                )
            )

        return False


# ============================================================================
# EVENT BUS
# ============================================================================

class EventBus:
    """
    Central event bus used throughout RENIX.

    Example:

        bus = EventBus()

        async def handler(event):
            print(event.name, event.data)

        bus.subscribe(
            "assistant.command",
            handler,
        )

        await bus.publish(
            "assistant.command",
            {
                "command": "hello"
            },
        )
    """

    def __init__(
        self,
        *,
        max_history: int = 1000,
        stop_on_error: bool = False,
    ) -> None:

        self.logger = logger

        self.max_history = max(
            1,
            int(max_history),
        )

        self.stop_on_error = bool(
            stop_on_error
        )

        self.listeners: dict[
            str,
            list[EventListener],
        ] = {}

        self.listener_index: dict[
            str,
            EventListener,
        ] = {}

        self.history: list[
            Event
        ] = []

        self.event_counts: dict[
            str,
            int,
        ] = {}

        self.error_counts: dict[
            str,
            int,
        ] = {}

        self.total_published = 0

        self.total_handled = 0

        self.total_errors = 0

        self.initialized = False

        self.started_at = time.time()

        self._lock = asyncio.Lock()

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the event bus.
        """

        self.initialized = True

        self.logger.info(
            "RENIX Event Bus initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the event bus and remove listeners.
        """

        async with self._lock:

            self.listeners.clear()

            self.listener_index.clear()

        self.initialized = False

        self.logger.info(
            "RENIX Event Bus shutdown."
        )

    # ========================================================================
    # SUBSCRIBE
    # ========================================================================

    def subscribe(
        self,
        event_name: str,
        handler: EventHandler,
        *,
        priority: int = 0,
        once: bool = False,
    ) -> str:
        """
        Register a handler for an event.

        Higher priority handlers execute first.
        """

        if not isinstance(
            event_name,
            str,
        ):

            raise TypeError(
                "event_name must be a string."
            )

        event_name = event_name.strip()

        if not event_name:

            raise ValueError(
                "event_name cannot be empty."
            )

        if not callable(handler):

            raise TypeError(
                "handler must be callable."
            )

        listener = EventListener(
            event_name=event_name,
            handler=handler,
            priority=int(priority),
            once=bool(once),
        )

        self.listeners.setdefault(
            event_name,
            [],
        ).append(
            listener
        )

        self.listener_index[
            listener.listener_id
        ] = listener

        self._sort_listeners(
            event_name
        )

        self.logger.debug(
            "Registered listener %s for %s",
            listener.listener_id,
            event_name,
        )

        return listener.listener_id

    # ========================================================================
    # ON
    # ========================================================================

    def on(
        self,
        event_name: str,
        handler: EventHandler,
        *,
        priority: int = 0,
    ) -> str:
        """
        Alias for subscribe().
        """

        return self.subscribe(
            event_name,
            handler,
            priority=priority,
            once=False,
        )

    # ========================================================================
    # ONCE
    # ========================================================================

    def once(
        self,
        event_name: str,
        handler: EventHandler,
        *,
        priority: int = 0,
    ) -> str:
        """
        Register a handler that executes only once.
        """

        return self.subscribe(
            event_name,
            handler,
            priority=priority,
            once=True,
        )

    # ========================================================================
    # ADD LISTENER
    # ========================================================================

    def add_listener(
        self,
        event_name: str,
        handler: EventHandler,
        *,
        priority: int = 0,
        once: bool = False,
    ) -> str:
        """
        Compatibility alias for subscribe().
        """

        return self.subscribe(
            event_name,
            handler,
            priority=priority,
            once=once,
        )

    # ========================================================================
    # SORT LISTENERS
    # ========================================================================

    def _sort_listeners(
        self,
        event_name: str,
    ) -> None:
        """
        Sort listeners by priority.
        """

        listeners = self.listeners.get(
            event_name
        )

        if not listeners:

            return

        listeners.sort(
            key=lambda listener:
            listener.priority,
            reverse=True,
        )

    # ========================================================================
    # UNSUBSCRIBE
    # ========================================================================

    def unsubscribe(
        self,
        listener_id: str,
    ) -> bool:
        """
        Remove a listener by ID.
        """

        listener = self.listener_index.get(
            listener_id
        )

        if listener is None:

            return False

        event_name = (
            listener.event_name
        )

        listeners = self.listeners.get(
            event_name,
            [],
        )

        self.listeners[
            event_name
        ] = [
            item
            for item in listeners
            if item.listener_id
            != listener_id
        ]

        self.listener_index.pop(
            listener_id,
            None,
        )

        if not self.listeners[
            event_name
        ]:

            self.listeners.pop(
                event_name,
                None,
            )

        self.logger.debug(
            "Removed listener %s",
            listener_id,
        )

        return True

    # ========================================================================
    # OFF
    # ========================================================================

    def off(
        self,
        listener_id: str,
    ) -> bool:
        """
        Alias for unsubscribe().
        """

        return self.unsubscribe(
            listener_id
        )

    # ========================================================================
    # REMOVE LISTENER
    # ========================================================================

    def remove_listener(
        self,
        listener_id: str,
    ) -> bool:
        """
        Compatibility alias for unsubscribe().
        """

        return self.unsubscribe(
            listener_id
        )

    # ========================================================================
    # REMOVE ALL
    # ========================================================================

    def remove_all_listeners(
        self,
        event_name: Optional[str] = None,
    ) -> int:
        """
        Remove listeners.

        If event_name is supplied, only listeners for
        that event are removed.

        Returns number of listeners removed.
        """

        if event_name is None:

            count = len(
                self.listener_index
            )

            self.listeners.clear()

            self.listener_index.clear()

            return count

        listeners = self.listeners.pop(
            event_name,
            [],
        )

        for listener in listeners:

            self.listener_index.pop(
                listener.listener_id,
                None,
            )

        return len(
            listeners
        )

    # ========================================================================
    # ENABLE LISTENER
    # ========================================================================

    def enable_listener(
        self,
        listener_id: str,
    ) -> bool:
        """
        Enable a listener.
        """

        listener = self.listener_index.get(
            listener_id
        )

        if listener is None:

            return False

        listener.enabled = True

        return True

    # ========================================================================
    # DISABLE LISTENER
    # ========================================================================

    def disable_listener(
        self,
        listener_id: str,
    ) -> bool:
        """
        Disable a listener without deleting it.
        """

        listener = self.listener_index.get(
            listener_id
        )

        if listener is None:

            return False

        listener.enabled = False

        return True

    # ========================================================================
    # CREATE EVENT
    # ========================================================================

    def create_event(
        self,
        event_name: str,
        data: Any = None,
        *,
        source: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Event:
        """
        Create an Event object.
        """

        if not isinstance(
            event_name,
            str,
        ):

            raise TypeError(
                "event_name must be a string."
            )

        event_name = event_name.strip()

        if not event_name:

            raise ValueError(
                "event_name cannot be empty."
            )

        return Event(
            name=event_name,
            data=data,
            source=source,
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

    # ========================================================================
    # PUBLISH
    # ========================================================================

    async def publish(
        self,
        event_name: str | Event,
        data: Any = None,
        *,
        source: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Event:
        """
        Publish an event.

        Supports:

            await bus.publish(
                "test",
                {"value": 1},
            )

        and:

            event = Event(...)
            await bus.publish(event)
        """

        if isinstance(
            event_name,
            Event,
        ):

            event = event_name

        else:

            event = self.create_event(
                event_name,
                data,
                source=source,
                metadata=metadata,
            )

        self.total_published += 1

        self.event_counts[
            event.name
        ] = (
            self.event_counts.get(
                event.name,
                0,
            )
            + 1
        )

        self._store_event(
            event
        )

        listeners = self._matching_listeners(
            event.name
        )

        for listener in listeners:

            if (
                event.propagation_stopped
            ):

                break

            if not listener.enabled:

                continue

            await self._execute_listener(
                listener,
                event,
            )

            if listener.once:

                self.unsubscribe(
                    listener.listener_id
                )

        return event

    # ========================================================================
    # EMIT
    # ========================================================================

    async def emit(
        self,
        event_name: str | Event,
        data: Any = None,
        *,
        source: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Event:
        """
        Alias for publish().
        """

        return await self.publish(
            event_name,
            data,
            source=source,
            metadata=metadata,
        )

    # ========================================================================
    # DISPATCH
    # ========================================================================

    async def dispatch(
        self,
        event: Event,
    ) -> Event:
        """
        Dispatch an already-created Event.
        """

        return await self.publish(
            event
        )

    # ========================================================================
    # MATCHING LISTENERS
    # ========================================================================

    def _matching_listeners(
        self,
        event_name: str,
    ) -> list[EventListener]:
        """
        Return all listeners matching an event.
        """

        matches: list[
            EventListener
        ] = []

        for listeners in self.listeners.values():

            for listener in listeners:

                if listener.matches(
                    event_name
                ):

                    matches.append(
                        listener
                    )

        matches.sort(
            key=lambda listener:
            listener.priority,
            reverse=True,
        )

        return matches

    # ========================================================================
    # EXECUTE LISTENER
    # ========================================================================

    async def _execute_listener(
        self,
        listener: EventListener,
        event: Event,
    ) -> Any:
        """
        Execute one listener safely.
        """

        listener.call_count += 1

        listener.last_called_at = (
            time.time()
        )

        self.total_handled += 1

        try:

            result = listener.handler(
                event
            )

            if inspect.isawaitable(
                result
            ):

                result = await result

            return result

        except asyncio.CancelledError:

            raise

        except Exception as exc:

            listener.error_count += 1

            listener.last_error = str(
                exc
            )

            self.total_errors += 1

            self.error_counts[
                event.name
            ] = (
                self.error_counts.get(
                    event.name,
                    0,
                )
                + 1
            )

            self.logger.exception(
                "Event handler failed: %s",
                event.name,
            )

            if self.stop_on_error:

                event.stop_propagation()

            return None

    # ========================================================================
    # STORE EVENT
    # ========================================================================

    def _store_event(
        self,
        event: Event,
    ) -> None:
        """
        Store an event in history.
        """

        self.history.append(
            event
        )

        if len(
            self.history
        ) > self.max_history:

            self.history = self.history[
                -self.max_history:
            ]

    # ========================================================================
    # GET HISTORY
    # ========================================================================

    def get_history(
        self,
        limit: Optional[int] = None,
        *,
        event_name: Optional[str] = None,
    ) -> list[Event]:
        """
        Return event history.
        """

        events = self.history

        if event_name is not None:

            events = [
                event
                for event in events
                if event.name
                == event_name
            ]

        if limit is None:

            return list(
                events
            )

        limit = max(
            0,
            int(limit),
        )

        if limit == 0:

            return []

        return events[
            -limit:
        ]

    # ========================================================================
    # CLEAR HISTORY
    # ========================================================================

    def clear_history(
        self,
    ) -> None:
        """
        Clear event history.
        """

        self.history.clear()

    # ========================================================================
    # GET LISTENERS
    # ========================================================================

    def get_listeners(
        self,
        event_name: Optional[str] = None,
    ) -> list[EventListener]:
        """
        Return registered listeners.
        """

        if event_name is not None:

            return list(
                self.listeners.get(
                    event_name,
                    [],
                )
            )

        return list(
            self.listener_index.values()
        )

    # ========================================================================
    # HAS LISTENERS
    # ========================================================================

    def has_listeners(
        self,
        event_name: str,
    ) -> bool:
        """
        Check whether an event has listeners.
        """

        return bool(
            self._matching_listeners(
                event_name
            )
        )

    # ========================================================================
    # WAIT FOR EVENT
    # ========================================================================

    async def wait_for(
        self,
        event_name: str,
        *,
        timeout: Optional[float] = None,
        predicate: Optional[
            Callable[[Event], bool]
        ] = None,
    ) -> Optional[Event]:
        """
        Wait until a matching event is published.

        Example:

            event = await bus.wait_for(
                "download.completed",
                timeout=30,
            )
        """

        loop = asyncio.get_running_loop()

        future: asyncio.Future[
            Event
        ] = loop.create_future()

        listener_id: Optional[str] = None

        async def handler(
            event: Event,
        ) -> None:

            if predicate is not None:

                try:

                    accepted = predicate(
                        event
                    )

                except Exception as exc:

                    self.logger.debug(
                        "Event predicate failed: %s",
                        exc,
                    )

                    return

                if not accepted:

                    return

            if not future.done():

                future.set_result(
                    event
                )

        listener_id = self.once(
            event_name,
            handler,
        )

        try:

            if timeout is None:

                return await future

            return await asyncio.wait_for(
                future,
                timeout=float(
                    timeout
                ),
            )

        except asyncio.TimeoutError:

            return None

        finally:

            if listener_id is not None:

                self.unsubscribe(
                    listener_id
                )

    # ========================================================================
    # WAIT FOR MULTIPLE EVENTS
    # ========================================================================

    async def wait_for_any(
        self,
        event_names: list[str],
        *,
        timeout: Optional[float] = None,
    ) -> Optional[Event]:
        """
        Wait for the first event from a list of event names.
        """

        if not event_names:

            return None

        loop = asyncio.get_running_loop()

        future: asyncio.Future[
            Event
        ] = loop.create_future()

        listener_ids: list[str] = []

        async def handler(
            event: Event,
        ) -> None:

            if not future.done():

                future.set_result(
                    event
                )

        try:

            for name in event_names:

                listener_ids.append(
                    self.once(
                        name,
                        handler,
                    )
                )

            if timeout is None:

                return await future

            return await asyncio.wait_for(
                future,
                timeout=float(
                    timeout
                ),
            )

        except asyncio.TimeoutError:

            return None

        finally:

            for listener_id in listener_ids:

                self.unsubscribe(
                    listener_id
                )

    # ========================================================================
    # EVENT COUNT
    # ========================================================================

    def get_event_count(
        self,
        event_name: str,
    ) -> int:
        """
        Return how many times an event has been published.
        """

        return self.event_counts.get(
            event_name,
            0,
        )

    # ========================================================================
    # ERROR COUNT
    # ========================================================================

    def get_error_count(
        self,
        event_name: Optional[str] = None,
    ) -> int:
        """
        Return event handler error count.
        """

        if event_name is None:

            return self.total_errors

        return self.error_counts.get(
            event_name,
            0,
        )

    # ========================================================================
    # RESET STATISTICS
    # ========================================================================

    def reset_statistics(
        self,
    ) -> None:
        """
        Reset event statistics.
        """

        self.event_counts.clear()

        self.error_counts.clear()

        self.total_published = 0

        self.total_handled = 0

        self.total_errors = 0

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return complete Event Bus statistics.
        """

        return {
            "initialized": (
                self.initialized
            ),
            "listener_count": len(
                self.listener_index
            ),
            "event_type_count": len(
                self.listeners
            ),
            "history_size": len(
                self.history
            ),
            "max_history": (
                self.max_history
            ),
            "total_published": (
                self.total_published
            ),
            "total_handled": (
                self.total_handled
            ),
            "total_errors": (
                self.total_errors
            ),
            "event_counts": dict(
                self.event_counts
            ),
            "error_counts": dict(
                self.error_counts
            ),
            "started_at": (
                self.started_at
            ),
        }

    # ========================================================================
    # DEBUG DUMP
    # ========================================================================

    def debug_dump(
        self,
    ) -> dict[str, Any]:
        """
        Return a detailed snapshot of the Event Bus.
        """

        listeners = []

        for listener in self.listener_index.values():

            listeners.append(
                {
                    "listener_id": (
                        listener.listener_id
                    ),
                    "event_name": (
                        listener.event_name
                    ),
                    "priority": (
                        listener.priority
                    ),
                    "once": (
                        listener.once
                    ),
                    "enabled": (
                        listener.enabled
                    ),
                    "call_count": (
                        listener.call_count
                    ),
                    "error_count": (
                        listener.error_count
                    ),
                    "last_called_at": (
                        listener.last_called_at
                    ),
                    "last_error": (
                        listener.last_error
                    ),
                }
            )

        return {
            "statistics": self.statistics(),
            "listeners": listeners,
            "history": [
                event.to_dict()
                for event in self.history
            ],
        }


# ============================================================================
# GLOBAL EVENT BUS
# ============================================================================

event_bus = EventBus()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def subscribe(
    event_name: str,
    handler: EventHandler,
    *,
    priority: int = 0,
    once: bool = False,
) -> str:
    """
    Register an event handler on the global Event Bus.
    """

    return event_bus.subscribe(
        event_name,
        handler,
        priority=priority,
        once=once,
    )


def on(
    event_name: str,
    handler: EventHandler,
    *,
    priority: int = 0,
) -> str:
    """
    Register a persistent global event handler.
    """

    return event_bus.on(
        event_name,
        handler,
        priority=priority,
    )


def once(
    event_name: str,
    handler: EventHandler,
    *,
    priority: int = 0,
) -> str:
    """
    Register a one-time global event handler.
    """

    return event_bus.once(
        event_name,
        handler,
        priority=priority,
    )


def unsubscribe(
    listener_id: str,
) -> bool:
    """
    Remove a global event handler.
    """

    return event_bus.unsubscribe(
        listener_id
    )


async def publish(
    event_name: str | Event,
    data: Any = None,
    *,
    source: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> Event:
    """
    Publish through the global Event Bus.
    """

    return await event_bus.publish(
        event_name,
        data,
        source=source,
        metadata=metadata,
    )


async def emit(
    event_name: str | Event,
    data: Any = None,
    *,
    source: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> Event:
    """
    Emit through the global Event Bus.
    """

    return await event_bus.emit(
        event_name,
        data,
        source=source,
        metadata=metadata,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Event",
    "EventListener",
    "EventBus",
    "event_bus",
    "subscribe",
    "on",
    "once",
    "unsubscribe",
    "publish",
    "emit",
]


