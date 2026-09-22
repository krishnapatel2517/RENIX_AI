"""
RENIX - Main Application Entry Point
====================================

Central startup file for the RENIX AI Assistant.

Responsibilities:
- Load environment variables
- Load configuration
- Initialize logging
- Initialize core services
- Initialize memory
- Initialize AI services
- Initialize voice system
- Initialize system monitoring
- Start RENIX
- Handle shutdown safely

Author: Krishna Patel
Project: RENIX
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import yaml


def configure_console_encoding() -> None:
    """Keep RENIX status output safe on Windows consoles."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(
                encoding="utf-8",
                errors="replace",
            )


configure_console_encoding()


def _desktop_ui_process(
    command_queue: Any,
    response_queue: Any,
) -> None:
    """Run the RENIX HUD dashboard entirely inside its own process."""

    import tkinter as tk
    from datetime import datetime

    colors = {
        "bg": "#020907",
        "panel": "#061813",
        "panel_alt": "#0a211a",
        "line": "#168b5c",
        "accent": "#69ff9b",
        "text": "#d8ffe4",
        "muted": "#78b892",
        "warning": "#f3d96b",
    }

    root = tk.Tk()
    root.title("RENIX | Personal Artificial Intelligence")
    root.geometry("1440x900")
    root.minsize(1100, 700)
    root.configure(bg=colors["bg"])

    def panel(parent: Any, title: str, row: int, column: int, **options: Any) -> Any:
        frame = tk.Frame(
            parent,
            bg=colors["panel"],
            highlightbackground=colors["line"],
            highlightthickness=1,
            bd=0,
            **options,
        )
        frame.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
        tk.Label(
            frame,
            text=title.upper(),
            bg=colors["panel"],
            fg=colors["accent"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(10, 6))
        content = tk.Frame(
            frame,
            bg=colors["panel"],
        )
        content.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        return content

    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)
    root.grid_columnconfigure(3, weight=1)

    header = tk.Frame(root, bg=colors["bg"])
    header.grid(row=0, column=0, columnspan=4, sticky="ew", padx=14, pady=12)
    header.grid_columnconfigure(1, weight=1)
    tk.Label(
        header,
        text="R  RENIX",
        bg=colors["bg"],
        fg=colors["accent"],
        font=("Segoe UI", 24, "bold"),
    ).grid(row=0, column=0, sticky="w")
    tk.Label(
        header,
        text="Your AI assistant  •  Always with you",
        bg=colors["bg"],
        fg=colors["muted"],
        font=("Segoe UI", 10),
    ).grid(row=1, column=0, sticky="w")
    status = tk.Label(
        header,
        text="● ONLINE   |   LISTENING READY",
        bg=colors["bg"],
        fg=colors["accent"],
        font=("Consolas", 11, "bold"),
    )
    status.grid(row=0, column=1, rowspan=2, sticky="e")

    nav = tk.Frame(root, bg=colors["panel"], highlightbackground=colors["line"], highlightthickness=1)
    nav.grid(row=1, column=0, sticky="nsew", padx=(14, 6), pady=6)
    tk.Label(nav, text="RENIX SYSTEMS", bg=colors["panel"], fg=colors["accent"], font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=14)

    for label, command in (
        ("⌂  Home", "status"),
        ("◉  Voice", "listen"),
        ("◌  Vision", "capabilities"),
        ("▣  Computer", "open Chrome"),
        ("▤  Files", "find my files"),
        ("◎  Browser", "search latest AI news"),
        ("◈  Research", "search latest AI news"),
        ("⌘  Coding", "diagnostics"),
        ("⚙  Automation", "capabilities"),
        ("▦  Devices", "capabilities"),
        ("▥  Education", "capabilities"),
        ("◉  Media", "capabilities"),
        ("⚙  Settings", "status"),
    ):
        button = tk.Button(
            nav,
            text=label,
            command=lambda value=command: command_queue.put(value),
            bg=colors["panel"],
            fg=colors["text"],
            activebackground=colors["panel_alt"],
            activeforeground=colors["accent"],
            relief=tk.FLAT,
            anchor="w",
            font=("Segoe UI", 10),
            padx=12,
            pady=6,
        )
        button.pack(fill=tk.X, padx=8, pady=1)

    status_panel = panel(root, "System status", 1, 1)
    for label, value in (("CPU", "12%"), ("GPU", "08%"), ("RAM", "34%"), ("DISK", "28%"), ("NETWORK", "CONNECTED"), ("VOICE", "READY")):
        row = tk.Frame(status_panel, bg=colors["panel"])
        row.pack(fill=tk.X, padx=12, pady=3)
        tk.Label(row, text=label, bg=colors["panel"], fg=colors["muted"], font=("Consolas", 9)).pack(side=tk.LEFT)
        tk.Label(row, text=value, bg=colors["panel"], fg=colors["accent"], font=("Consolas", 10, "bold")).pack(side=tk.RIGHT)

    center = panel(root, "RENIX core", 1, 2)
    center.grid_columnconfigure(0, weight=1)
    tk.Label(center, text="◉", bg=colors["panel"], fg=colors["accent"], font=("Segoe UI", 96, "bold")).pack(pady=(28, 0))
    tk.Label(center, text="Hey. I am RENIX.", bg=colors["panel"], fg=colors["accent"], font=("Segoe UI", 16, "bold")).pack()
    tk.Label(center, text="What would you like me to do today?", bg=colors["panel"], fg=colors["text"], font=("Segoe UI", 10)).pack(pady=(2, 18))
    waveform = tk.Label(center, text="▁▂▃▅▇▅▃▂▁▂▅▇▅▂▁", bg=colors["panel"], fg=colors["accent"], font=("Consolas", 18))
    waveform.pack(pady=10)

    activity = panel(root, "Recent activity", 1, 3)
    for item in ("Opened Chrome", "Memory online", "Voice services ready", "Vision manager ready", "Security active"):
        tk.Label(activity, text=f"●  {item}", bg=colors["panel"], fg=colors["text"], anchor="w", font=("Segoe UI", 9)).pack(fill=tk.X, padx=12, pady=5)

    quick = panel(root, "Quick actions", 2, 1)
    quick.grid_columnconfigure(0, weight=1)
    quick.grid_columnconfigure(1, weight=1)
    for index, (label, command) in enumerate((("Open Chrome", "open Chrome"), ("Find my files", "find my files"), ("Take screenshot", "take a screenshot"), ("Listen", "listen"))):
        tk.Button(quick, text=label, command=lambda value=command: command_queue.put(value), bg=colors["panel_alt"], fg=colors["accent"], activebackground=colors["line"], relief=tk.FLAT, font=("Segoe UI", 9), padx=8, pady=6).grid(row=index // 2, column=index % 2, sticky="ew", padx=6, pady=5)

    context = panel(root, "Memory & context", 2, 2)
    tk.Label(context, text="Recalling memories...", bg=colors["panel"], fg=colors["muted"], anchor="w").pack(fill=tk.X, padx=12, pady=5)
    tk.Label(context, text="Project: RENIX", bg=colors["panel"], fg=colors["text"], anchor="w").pack(fill=tk.X, padx=12, pady=5)
    tk.Label(context, text="Current task: Ready", bg=colors["panel"], fg=colors["accent"], anchor="w").pack(fill=tk.X, padx=12, pady=5)

    activity_log = panel(root, "Notifications", 2, 3)
    for item in ("System update available", "Voice command ready", "Reminder: Explore RENIX"):
        tk.Label(activity_log, text=item, bg=colors["panel"], fg=colors["text"], anchor="w").pack(fill=tk.X, padx=12, pady=5)

    command_bar = tk.Frame(root, bg=colors["bg"])
    command_bar.grid(row=3, column=0, columnspan=4, sticky="ew", padx=14, pady=(8, 14))
    command_bar.grid_columnconfigure(0, weight=1)
    entry = tk.Entry(command_bar, bg=colors["panel_alt"], fg=colors["text"], insertbackground=colors["accent"], relief=tk.FLAT, font=("Segoe UI", 12))
    entry.grid(row=0, column=0, sticky="ew", ipady=10, padx=(0, 8))
    entry.insert(0, "Ask me anything...")

    def submit(_event: Any = None) -> None:
        text = entry.get().strip()
        if not text or text == "Ask me anything...":
            return
        entry.delete(0, tk.END)
        command_queue.put(text)

    entry.bind("<Return>", submit)
    tk.Button(command_bar, text="➤", command=submit, bg=colors["accent"], fg=colors["bg"], relief=tk.FLAT, font=("Segoe UI", 13, "bold"), width=4).grid(row=0, column=1, ipady=6)

    footer = tk.Label(root, text="RENIX  •  Think  •  Plan  •  Execute  •  Verify  •  For a Better You", bg=colors["bg"], fg=colors["muted"], font=("Consolas", 8))
    footer.grid(row=4, column=0, columnspan=4, pady=(0, 8))

    def poll_responses() -> None:
        while not response_queue.empty():
            status.configure(text=f"● ONLINE   |   {response_queue.get().upper()}")
        root.after(100, poll_responses)

    root.after(100, poll_responses)

    def close() -> None:
        command_queue.put(None)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


class _DesktopUI:
    """Small native Windows fallback UI for the RENIX command loop."""

    def __init__(self, app: "RENIX") -> None:
        self.app = app
        self._process: Any = None
        self._bridge_thread: Any = None
        self._commands: Any = None
        self._responses: Any = None

    def start(self) -> bool:
        import threading

        context = multiprocessing.get_context("spawn")
        self._commands = context.Queue()
        self._responses = context.Queue()
        self._process = context.Process(
            target=_desktop_ui_process,
            args=(self._commands, self._responses),
            name="RENIX-NativeUI",
            daemon=True,
        )
        self._process.start()

        def bridge() -> None:
            while self._process is not None and self._process.is_alive():
                command = self._commands.get()
                if command is None:
                    break
                future = asyncio.run_coroutine_threadsafe(
                    self.app.handle_command(command),
                    self.app._loop,
                )
                try:
                    future.result()
                    self._responses.put("Command processed.")
                except Exception as error:
                    self._responses.put(f"Command failed: {error}")

        self._bridge_thread = threading.Thread(
            target=bridge,
            name="RENIX-UI-Bridge",
            daemon=True,
        )
        self._bridge_thread.start()
        return bool(self._process.is_alive())

    def stop(self) -> None:
        if self._commands is not None:
            self._commands.put(None)
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        if self._bridge_thread is not None and self._bridge_thread.is_alive():
            self._bridge_thread.join(timeout=1.0)


class _VoiceRuntime:
    """Connect microphone, speech recognition, and speech output."""

    def __init__(self, orchestrator: Any) -> None:
        from voice.microphone import MicrophoneManager
        from voice.language_detection import LanguageDetector
        from voice.speech_to_text import STTConfig, SpeechToText
        from voice.text_to_speech import TTSConfig, TextToSpeech

        self.microphone = MicrophoneManager()
        language = os.getenv(
            "RENIX_SPEECH_LANGUAGE",
            "en-IN",
        )
        self.stt = SpeechToText(
            config=STTConfig(language=language),
            microphone=self.microphone,
        )
        self.tts = TextToSpeech(
            config=TTSConfig(language=language.split("-")[0]),
        )
        self.language_detection = LanguageDetector()
        self.orchestrator = orchestrator
        self._stop_event = threading.Event()
        self._thread: Any = None
        self._loop: Any = None

    @property
    def listening_available(self) -> bool:
        return bool(self.microphone.available and self.stt.available)

    @property
    def speaking_available(self) -> bool:
        return bool(self.tts.available)

    def listen_once(self) -> Any:
        result = self.stt.listen()
        if result.success and result.text:
            detected = self.language_detection.detect(result.text)
            language = (
                detected.language.code
                if detected.language is not None
                else None
            )
            return self.orchestrator.handle_text(
                result.text,
                source="voice",
                metadata={
                    "language": language,
                    "language_confidence": detected.confidence,
                },
            )
        return result

    def speak(self, text: str, language: str | None = None) -> Any:
        return self.tts.speak_sync(
            text,
            language=language,
        )

    def start_continuous(self, loop: Any) -> bool:
        """Listen continuously in a worker thread and route every utterance."""

        if not self.listening_available or self._thread is not None:
            return False

        self._loop = loop
        self._stop_event.clear()

        def worker() -> None:
            while not self._stop_event.is_set():
                try:
                    result = self.stt.listen(
                        timeout=2.0,
                        phrase_time_limit=15.0,
                    )
                    if not result.success or not result.text:
                        continue

                    detected = self.language_detection.detect(result.text)
                    language = (
                        detected.language.code
                        if detected.language is not None
                        else None
                    )
                    future = asyncio.run_coroutine_threadsafe(
                        self.orchestrator.handle_text(
                            result.text,
                            source="voice",
                            metadata={
                                "language": language,
                                "language_confidence": detected.confidence,
                            },
                        ),
                        self._loop,
                    )
                    response = future.result()
                    if (
                        self.speaking_available
                        and getattr(response, "message", "")
                    ):
                        self.speak(response.message, language)
                except Exception as error:
                    logger.debug("Continuous voice cycle failed: %s", error)

        self._thread = threading.Thread(
            target=worker,
            name="RENIX-ContinuousVoice",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop_continuous(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def stop(self) -> None:
        self.stop_continuous()
        stop = getattr(self.microphone, "stop", None)
        if callable(stop):
            stop()
        stop_tts = getattr(self.tts, "stop", None)
        if callable(stop_tts):
            stop_tts()


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# OPTIONAL DOTENV
# ============================================================

try:
    from dotenv import load_dotenv

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

except ImportError:
    pass


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> logging.Logger:
    """
    Configure RENIX logging.
    """

    log_directory = (
        PROJECT_ROOT
        / "logs"
        / "system"
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        log_directory
        / "renix.log"
    )

    logger = logging.getLogger(
        "RENIX"
    )

    logger.setLevel(
        logging.INFO
    )

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    console_handler = (
        logging.StreamHandler()
    )

    console_handler.setFormatter(
        formatter
    )

    file_handler = (
        logging.FileHandler(
            log_file,
            encoding="utf-8",
        )
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    return logger


logger = setup_logging()


class _AIProviderAdapter:
    """Expose the provider layer through the orchestrator's chat API."""

    def __init__(self, provider: Any, system_prompt: str) -> None:
        self.provider = provider
        self.system_prompt = system_prompt

    def chat(self, text: str) -> str:
        from integrations.providers.ai.provider import AIRequest

        response = self.provider.generate(
            AIRequest(
                prompt=text,
                system_prompt=self.system_prompt,
                model=self.provider.default_model,
            )
        )
        return response.text

    def shutdown(self) -> None:
        shutdown = getattr(self.provider, "shutdown", None)
        if callable(shutdown):
            shutdown()


# ============================================================
# RENIX APPLICATION
# ============================================================

class RENIX:
    """
    Main RENIX AI Assistant application.

    This class acts as the central controller
    connecting all RENIX modules.
    """

    def __init__(self) -> None:

        self.running = False

        self.services: dict[
            str,
            Any
        ] = {}

        self.config: dict[
            str,
            Any
        ] = {}

        self.startup_errors: list[
            str
        ] = []

        self._loop: Any = None

        logger.info(
            "RENIX application instance created."
        )

    # ========================================================
    # STARTUP
    # ========================================================

    async def start(
        self,
    ) -> None:
        """
        Start the complete RENIX system.
        """

        if self.running:

            logger.warning(
                "RENIX is already running."
            )

            return

        self.print_banner()

        logger.info(
            "Starting RENIX..."
        )

        try:

            await self.initialize()

            self.running = True

            logger.info(
                "RENIX started successfully."
            )

            print(
                "\n"
                "╔══════════════════════════════════════╗"
            )

            print(
                "║          RENIX IS ONLINE             ║"
            )

            print(
                "╚══════════════════════════════════════╝"
                "\n"
            )

            await self.run()

        except KeyboardInterrupt:

            logger.info(
                "Keyboard interrupt received."
            )

        except Exception as error:

            logger.exception(
                "Critical RENIX startup error: %s",
                error,
            )

            print(
                f"\n[RENIX ERROR] {error}"
            )

        finally:

            await self.shutdown()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize all RENIX components.
        """

        logger.info(
            "Initializing RENIX systems..."
        )

        self._loop = asyncio.get_running_loop()

        await self.load_configuration()

        await self.initialize_directories()

        await self.initialize_database()

        await self.initialize_core()

        await self.initialize_memory()

        await self.initialize_ai()

        await self.initialize_system_monitor()

        await self.initialize_voice()

        await self.initialize_security()

        await self.initialize_computer()

        await self.initialize_optional_agents()

        await self.initialize_domain_agents()

        await self.initialize_vision()

        await self.initialize_ui()

        await self.initialize_education()

        await self.initialize_personal()

        logger.info(
            "RENIX initialization complete."
        )

        if self.startup_errors:

            logger.warning(
                "%s component(s) failed "
                "to initialize.",
                len(
                    self.startup_errors
                ),
            )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    async def load_configuration(
        self,
    ) -> None:
        """
        Load RENIX configuration.
        """

        logger.info(
            "Loading configuration..."
        )

        try:
            configuration = yaml.safe_load(
                (PROJECT_ROOT / "config.yaml").read_text(
                    encoding="utf-8"
                )
            ) or {}
        except (OSError, yaml.YAMLError) as error:
            raise RuntimeError(
                f"Unable to load config.yaml: {error}"
            ) from error

        renix_config = configuration.get("renix", {})
        voice_config = configuration.get("voice", {})
        ui_config = configuration.get("ui", {})

        self.config = {
            "master": configuration,
            "name": os.getenv(
                "RENIX_NAME",
                renix_config.get("name", "RENIX"),
            ),
            "version": os.getenv(
                "RENIX_VERSION",
                renix_config.get("version", "1.0.0"),
            ),
            "debug": os.getenv(
                "RENIX_DEBUG",
                str(renix_config.get("debug", False)),
            ).lower() == "true",
            "voice_enabled": os.getenv(
                "RENIX_VOICE_ENABLED",
                str(voice_config.get("enabled", True)),
            ).lower() == "true",
            "ui_enabled": os.getenv(
                "RENIX_UI_ENABLED",
                str(ui_config.get("enabled", True)),
            ).lower() == "true",
            "mode": os.getenv(
                "RENIX_MODE",
                "text",
            ).strip().lower() or "text",
        }

        logger.info(
            "Configuration loaded."
        )

    # ========================================================
    # DIRECTORIES
    # ========================================================

    async def initialize_directories(
        self,
    ) -> None:
        """
        Create required RENIX directories.
        """

        logger.info(
            "Checking project directories..."
        )

        directories = [

            "data/memory",
            "data/conversations",
            "data/projects",
            "data/tasks",
            "data/embeddings",
            "data/profiles",
            "data/statistics",
            "data/cache",
            "data/logs",
            "data/backups",

            "logs/system",
            "logs/ai",
            "logs/security",
            "logs/automation",
            "logs/errors",
            "logs/audit",

            "assets/icons",
            "assets/images",
            "assets/models",
            "assets/textures",
            "assets/animations",
            "assets/sounds",
            "assets/music",
            "assets/fonts",
            "assets/shaders",
        ]

        for directory in directories:

            path = (
                PROJECT_ROOT
                / directory
            )

            path.mkdir(
                parents=True,
                exist_ok=True,
            )

        logger.info(
            "Directories initialized."
        )

    # ========================================================
    # DATABASE
    # ========================================================

    async def initialize_database(
        self,
    ) -> None:
        """Initialize the shared SQLite database and schema."""

        try:
            from database import create_database
            from database.database import DatabaseConfig

            database_path = PROJECT_ROOT / "data" / "renix.db"
            database_exists = database_path.exists()
            database = create_database(
                config=DatabaseConfig(
                    path=database_path
                )
            )
            schema_file = PROJECT_ROOT / "database" / "schema.sql"
            schema_sql = (
                schema_file.read_text(encoding="utf-8")
                if schema_file.exists() and not database_exists
                else None
            )
            database.initialize(
                schema_sql=schema_sql
            )
            self.services["database"] = database
            logger.info("Database initialized.")

        except Exception as error:
            self.register_startup_error("Database", error)

    # ========================================================
    # CORE
    # ========================================================

    async def initialize_core(
        self,
    ) -> None:
        """
        Initialize core RENIX systems.
        """

        try:

            from core.orchestrator import (
                Orchestrator,
            )

            self.services[
                "orchestrator"
            ] = Orchestrator()

            await self.services[
                "orchestrator"
            ].initialize()

            logger.info(
                "Core orchestrator initialized."
            )

        except Exception as error:

            self.register_startup_error(
                "Core",
                error,
            )

    # ========================================================
    # MEMORY
    # ========================================================

    async def initialize_memory(
        self,
    ) -> None:
        """
        Initialize RENIX memory.
        """

        try:

            from memory.memory_manager import (
                MemoryManager,
            )

            memory = MemoryManager(
                storage_path=str(
                    PROJECT_ROOT
                    / "data"
                    / "memory"
                    / "memories.json"
                )
            )

            self.services[
                "memory"
            ] = memory

            orchestrator = self.services.get("orchestrator")
            if orchestrator is not None:
                orchestrator.memory = memory

            logger.info(
                "Memory system initialized."
            )

        except Exception as error:

            self.register_startup_error(
                "Memory",
                error,
            )

    # ========================================================
    # AI
    # ========================================================

    async def initialize_ai(
        self,
    ) -> None:
        """
        Initialize AI systems.
        """

        try:

            from ai.personality.personality import (
                Personality,
            )

            personality = Personality()

            self.services[
                "personality"
            ] = personality

            provider_name = os.getenv(
                "RENIX_DEFAULT_AI_PROVIDER",
                "gemini",
            ).strip().lower()
            provider_key_names = {
                "openai": "OPENAI_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
            }
            provider_key = os.getenv(
                provider_key_names.get(provider_name, ""),
                "",
            ).strip()

            if provider_key:
                from integrations.providers.ai import discover_provider

                provider_class = discover_provider(provider_name)
                if provider_class is None:
                    raise RuntimeError(
                        f"AI provider '{provider_name}' is unavailable."
                    )

                provider = provider_class()
                provider.initialize()
                provider_adapter = _AIProviderAdapter(
                    provider,
                    self._configuration_value(
                        "master.ai.system_prompt",
                        "You are RENIX, a helpful and safety-aware assistant.",
                    ),
                )
                self.services["ai_provider"] = provider_adapter

                orchestrator = self.services.get("orchestrator")
                if orchestrator is not None:
                    orchestrator.ai = provider_adapter

                logger.info(
                    "AI provider '%s' initialized.",
                    provider_name,
                )
            else:
                logger.warning(
                    "No API key configured for AI provider '%s'. "
                    "Continuing in offline mode.",
                    provider_name,
                )

            logger.info(
                "AI personality initialized."
            )

        except Exception as error:

            self.register_startup_error(
                "AI Personality",
                error,
            )

    def _configuration_value(
        self,
        path: str,
        default: Any,
    ) -> Any:
        """Read a dotted value from the loaded master configuration."""

        value: Any = self.config
        for part in path.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
        return default if value is None else value

    # ========================================================
    # SYSTEM MONITOR
    # ========================================================

    async def initialize_system_monitor(
        self,
    ) -> None:
        """
        Initialize system monitoring.
        """

        try:

            from system_monitor.monitor import (
                SystemMonitor,
            )

            monitor = SystemMonitor()

            self.services[
                "system_monitor"
            ] = monitor

            logger.info(
                "System monitor initialized."
            )

        except Exception as error:

            self.register_startup_error(
                "System Monitor",
                error,
            )

    # ========================================================
    # VOICE
    # ========================================================

    async def initialize_voice(
        self,
    ) -> None:
        """
        Initialize voice services.
        """

        if not self.config.get(
            "voice_enabled",
            True,
        ):
            logger.info(
                "Voice system disabled."
            )

            return

        try:

            from voice.conversation_mode import (
                ConversationMode,
            )

            conversation = (
                ConversationMode()
            )

            self.services[
                "voice"
            ] = conversation

            orchestrator = self.services.get("orchestrator")
            if orchestrator is None:
                raise RuntimeError(
                    "Voice services require the orchestrator."
                )

            voice_runtime = _VoiceRuntime(orchestrator)
            self.services["voice_runtime"] = voice_runtime

            auto_listen = os.getenv(
                "RENIX_VOICE_AUTO_LISTEN",
                "true",
            ).strip().lower() == "true"
            continuous_started = (
                voice_runtime.start_continuous(self._loop)
                if auto_listen
                else False
            )

            logger.info(
                "Voice services initialized: microphone=%s, stt=%s, tts=%s, continuous=%s.",
                voice_runtime.microphone.available,
                voice_runtime.stt.available,
                voice_runtime.tts.available,
                continuous_started,
            )

            logger.info(
                "Voice system initialized."
            )

        except Exception as error:

            self.register_startup_error(
                "Voice",
                error,
            )

    async def initialize_ui(
        self,
    ) -> None:
        """Start the native UI when enabled, with a safe headless fallback."""

        if not self.config.get("ui_enabled", True):
            logger.info("UI disabled by configuration.")
            return

        try:
            ui = _DesktopUI(self)
            if ui.start():
                self.services["ui"] = ui
                logger.info("Native RENIX UI started.")
            else:
                logger.warning("UI unavailable; continuing in terminal mode.")
        except Exception as error:
            self.register_startup_error("UI", error)

    # ========================================================
    # SECURITY
    # ========================================================

    async def initialize_security(
        self,
    ) -> None:
        """
        Initialize security services.
        """

        try:

            from security.security_manager import (
                SecurityManager,
            )

            security = (
                SecurityManager()
            )

            self.services[
                "security"
            ] = security

            logger.info(
                "Security system initialized."
            )

        except Exception as error:

            self.register_startup_error(
                "Security",
                error,
            )

    async def initialize_computer(
        self,
    ) -> None:
        """Initialize the computer agent and connect its actions."""

        try:
            from ai.agents.base_agent import AgentTask
            from ai.agents.computer_agent import ComputerAgent

            agent = ComputerAgent()
            self.services["computer_agent"] = agent

            orchestrator = self.services.get("orchestrator")
            if orchestrator is None:
                raise RuntimeError(
                    "Computer agent requires the orchestrator."
                )

            action_types = (
                "open_application",
                "close_application",
                "screenshot",
                "take_screenshot",
                "clipboard_get",
                "clipboard_set",
                "screen_info",
                "launch_url",
                "open_file",
            )

            async def execute_computer_action(action):
                task_parameters = dict(action.parameters)
                task_parameters.setdefault("action", action.action_type)
                task = AgentTask.create(
                    action.action_type.replace("_", " "),
                    parameters=task_parameters,
                    metadata={"request_id": action.request_id},
                )
                result = await agent.run(task)
                if hasattr(result, "to_dict"):
                    return result.to_dict()
                return result

            for action_type in action_types:
                orchestrator.register_action(
                    action_type,
                    execute_computer_action,
                )

            logger.info("Computer agent initialized and connected.")

        except Exception as error:
            self.register_startup_error("Computer", error)

    async def initialize_optional_agents(
        self,
    ) -> None:
        """Connect existing file and browser agents when available."""

        try:
            from ai.agents.base_agent import AgentTask
            from ai.agents.browser_agent import BrowserAgent
            from ai.agents.file_agent import FileAgent

            orchestrator = self.services.get("orchestrator")
            if orchestrator is None:
                raise RuntimeError(
                    "Optional agents require the orchestrator."
                )

            agents = {
                "file_agent": (FileAgent(), ("search_files", "read_file")),
                "browser_agent": (
                    BrowserAgent(),
                    ("search_web", "open_url", "read_page"),
                ),
            }

            for name, (agent, action_types) in agents.items():
                self.services[name] = agent

                async def execute_agent_action(
                    action,
                    current_agent=agent,
                ):
                    task_parameters = dict(action.parameters)
                    task_parameters.setdefault("action", action.action_type)
                    task = AgentTask.create(
                        action.action_type.replace("_", " "),
                        parameters=task_parameters,
                        metadata={"request_id": action.request_id},
                    )
                    result = await current_agent.run(task)
                    if hasattr(result, "to_dict"):
                        return result.to_dict()
                    return result

                for action_type in action_types:
                    orchestrator.register_action(
                        action_type,
                        execute_agent_action,
                    )

            logger.info("File and browser agents initialized.")

        except Exception as error:
            self.register_startup_error("Optional Agents", error)

    async def initialize_domain_agents(
        self,
    ) -> None:
        """Register the existing domain agents behind one safe dispatcher."""

        try:
            from ai.agents.base_agent import AgentTask
            from ai.agents.automation_agent import AutomationAgent
            from ai.agents.coding_agent import CodingAgent
            from ai.agents.cricket_agent import CricketAgent
            from ai.agents.device_agent import DeviceAgent
            from ai.agents.research_agent import ResearchAgent
            from ai.agents.robotics_agent import RoboticsAgent
            from ai.agents.study_agent import StudyAgent
            from ai.agents.system_agent import SystemAgent

            orchestrator = self.services.get("orchestrator")
            if orchestrator is None:
                raise RuntimeError(
                    "Domain agents require the orchestrator."
                )

            agents = {
                "coding_agent": CodingAgent(),
                "research_agent": ResearchAgent(),
                "study_agent": StudyAgent(),
                "cricket_agent": CricketAgent(),
                "device_agent": DeviceAgent(),
                "robotics_agent": RoboticsAgent(),
                "system_agent": SystemAgent(),
                "automation_agent": AutomationAgent(),
            }

            async def execute_domain_action(action):
                agent_name = str(
                    action.parameters.get("agent", "")
                ).strip().lower()
                agent = agents.get(agent_name)
                if agent is None:
                    return {
                        "success": False,
                        "error": f"Domain agent unavailable: {agent_name}",
                    }

                instruction = str(
                    action.parameters.get(
                        "instruction",
                        action.action_type,
                    )
                )
                parameters = dict(action.parameters)
                parameters.pop("agent", None)
                parameters.pop("instruction", None)

                if hasattr(agent, "run"):
                    result = await agent.run(
                        AgentTask.create(
                            instruction,
                            parameters=parameters,
                            metadata={"request_id": action.request_id},
                        )
                    )
                else:
                    execute = getattr(agent, "execute", None)
                    if not callable(execute):
                        return {
                            "success": False,
                            "error": f"Agent has no execution API: {agent_name}",
                        }
                    result = execute(
                        action="request",
                        parameters={
                            **parameters,
                            "instruction": instruction,
                        },
                    )
                    if asyncio.iscoroutine(result):
                        result = await result

                return result.to_dict() if hasattr(result, "to_dict") else result

            for agent_name, agent in agents.items():
                self.services[agent_name] = agent
            orchestrator.register_action(
                "agent_request",
                execute_domain_action,
            )
            logger.info(
                "Domain agents initialized: %s.",
                ", ".join(sorted(agents)),
            )

        except Exception as error:
            self.register_startup_error("Domain Agents", error)

    async def initialize_vision(
        self,
    ) -> None:
        """Register the lazy camera manager without requiring hardware."""

        if not self._configuration_value(
            "master.vision.enabled",
            True,
        ):
            logger.info("Vision system disabled by configuration.")
            return

        try:
            from vision.camera import CameraConfig
            from vision.camera_manager import CameraManager

            camera_config = CameraConfig(
                device_index=int(
                    self._configuration_value(
                        "master.vision.camera.index",
                        0,
                    )
                ),
                width=int(
                    self._configuration_value(
                        "master.vision.camera.width",
                        1280,
                    )
                ),
                height=int(
                    self._configuration_value(
                        "master.vision.camera.height",
                        720,
                    )
                ),
                fps=int(
                    self._configuration_value(
                        "master.vision.camera.fps",
                        30,
                    )
                ),
            )
            self.services["vision"] = CameraManager(
                auto_discover=False,
                default_config=camera_config,
            )
            logger.info(
                "Vision manager initialized in lazy hardware mode."
            )

        except Exception as error:
            self.register_startup_error("Vision", error)

    # ========================================================
    # EDUCATION
    # ========================================================

    async def initialize_education(
        self,
    ) -> None:
        """
        Initialize education services.
        """

        try:

            from education.study_manager import (
                StudyManager,
            )

            study_manager = (
                StudyManager()
            )

            self.services[
                "education"
            ] = study_manager

            logger.info(
                "Education system initialized."
            )

        except Exception as error:

            self.register_startup_error(
                "Education",
                error,
            )

    # ========================================================
    # PERSONAL
    # ========================================================

    async def initialize_personal(
        self,
    ) -> None:
        """
        Initialize personal productivity tools.
        """

        try:

            from personal.tasks import (
                TaskManager,
            )

            task_manager = (
                TaskManager()
            )

            self.services[
                "tasks"
            ] = task_manager

            logger.info(
                "Personal task system initialized."
            )

        except Exception as error:

            self.register_startup_error(
                "Personal",
                error,
            )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    async def run(
        self,
    ) -> None:
        """
        Main RENIX runtime loop.

        Currently provides a terminal interface.
        Later this will connect to:
        - Voice input
        - Holographic UI
        - Gesture control
        - Computer automation
        """

        logger.info(
            "Entering RENIX main loop."
        )

        print(
            "Type 'help' for commands."
        )

        print(
            "Type 'exit' to shut down RENIX.\n"
        )

        while self.running:

            try:

                command = await asyncio.to_thread(
                    input,
                    "You > ",
                )

                command = command.strip()

                if not command:
                    continue

                await self.handle_command(
                    command
                )

            except EOFError:

                break

            except KeyboardInterrupt:

                break

            except Exception as error:

                logger.exception(
                    "Command processing error: %s",
                    error,
                )

                print(
                    f"RENIX > Error: {error}"
                )

    # ========================================================
    # COMMAND HANDLER
    # ========================================================

    async def handle_command(
        self,
        command: str,
    ) -> None:
        """
        Process user commands.
        """

        normalized = command.lower().strip()

        if normalized in {
            "exit",
            "quit",
            "shutdown",
            "stop",
        }:

            self.running = False

            print(
                "RENIX > Shutting down..."
            )

            return

        if normalized == "help":

            self.show_help()

            return

        if normalized in {
            "status",
            "system status",
        }:

            self.show_status()

            return

        if normalized == "services":

            self.show_services()

            return

        if normalized in {"capabilities", "capability", "features"}:
            self.show_capabilities()
            return

        if normalized in {"errors", "startup errors", "diagnostics"}:
            self.show_startup_errors()
            return

        if normalized in {"voice", "listen", "listen mode", "voice mode"}:
            await self.listen_once()
            return

        orchestrator = self.services.get(
            "orchestrator"
        )

        if orchestrator:

            try:

                if hasattr(
                    orchestrator,
                    "handle_text",
                ):

                    response = await orchestrator.handle_text(
                        command,
                        source="text",
                    )

                    if hasattr(response, "message"):
                        message = response.message
                        voice_runtime = self.services.get("voice_runtime")
                        if (
                            voice_runtime is not None
                            and getattr(
                                voice_runtime,
                                "speaking_available",
                                False,
                            )
                            and message
                        ):
                            await asyncio.to_thread(
                                voice_runtime.speak,
                                message,
                            )
                        response = message

                    print(
                        f"RENIX > {response}"
                    )

                    return

            except Exception as error:

                logger.error(
                    "Orchestrator error: %s",
                    error,
                )

        print(
            "RENIX > I received your command: "
            f"{command}"
        )

        print(
            "RENIX > Advanced command routing "
            "will handle this automatically."
        )

    # ========================================================
    # HELP
    # ========================================================

    def show_help(
        self,
    ) -> None:

        print(
            "\n"
            "════════ RENIX COMMANDS ════════\n"
        )

        commands = {
            "help":
                "Show available commands",

            "status":
                "Show RENIX system status",

            "services":
                "Show initialized services",

            "capabilities":
                "Show available and degraded capabilities",

            "diagnostics":
                "Show startup warnings and errors",

            "listen":
                "Listen for one voice command",

            "hello":
                "Send a greeting through the assistant",

            "exit":
                "Shutdown RENIX",
        }

        for command, description in (
            commands.items()
        ):

            print(
                f"  {command:<12} "
                f"- {description}"
            )

        print()

    async def listen_once(
        self,
    ) -> None:
        """Capture and route one voice command when hardware is available."""

        runtime = self.services.get("voice_runtime")
        if runtime is None or not runtime.listening_available:
            print(
                "RENIX > Voice input is unavailable. "
                "Check microphone access and SpeechRecognition setup."
            )
            return

        print("RENIX > Listening...")
        try:
            result = await asyncio.to_thread(runtime.listen_once)
            if hasattr(result, "message"):
                print(f"RENIX > {result.message}")
            elif hasattr(result, "text"):
                print(f"RENIX > Heard: {result.text}")
            else:
                print(f"RENIX > Voice result: {result}")
        except Exception as error:
            logger.warning("Voice command failed: %s", error)
            print(f"RENIX > Voice input failed: {error}")

    # ========================================================
    # STATUS
    # ========================================================

    def show_status(
        self,
    ) -> None:

        print(
            "\n"
            "════════ RENIX STATUS ════════"
        )

        print(
            f"Name: {self.config.get('name')}"
        )

        print(
            f"Version: "
            f"{self.config.get('version')}"
        )

        print(
            f"Running: {self.running}"
        )

        print(
            f"Mode: {self.config.get('mode', 'text')}"
        )

        print(
            f"Services: "
            f"{len(self.services)}"
        )

        print(
            f"Startup Errors: "
            f"{len(self.startup_errors)}"
        )

        print(
            "AI: "
            + (
                "connected"
                if "ai_provider" in self.services
                else "offline"
            )
        )

        print(
            "Vision: "
            + (
                "ready"
                if "vision" in self.services
                else "unavailable"
            )
        )

        voice_runtime = self.services.get("voice_runtime")
        print(
            "Voice input: "
            + (
                "ready"
                if voice_runtime is not None
                and voice_runtime.listening_available
                else "unavailable"
            )
        )
        print(
            "Voice output: "
            + (
                "ready"
                if voice_runtime is not None
                and voice_runtime.speaking_available
                else "unavailable"
            )
        )

        print(
            "══════════════════════════════\n"
        )

    def show_capabilities(
        self,
    ) -> None:
        """Display user-facing capability availability."""

        capabilities = {
            "Text commands": "available",
            "Memory": "available" if "memory" in self.services else "unavailable",
            "Computer control": "available" if "computer_agent" in self.services else "unavailable",
            "File search": "available" if "file_agent" in self.services else "unavailable",
            "Browser actions": "available" if "browser_agent" in self.services else "unavailable",
            "Voice input": (
                "ready"
                if self.services.get("voice_runtime") is not None
                and self.services["voice_runtime"].listening_available
                else "unavailable"
            ),
            "Voice output": (
                "ready"
                if self.services.get("voice_runtime") is not None
                and self.services["voice_runtime"].speaking_available
                else "unavailable"
            ),
            "Vision": "available" if "vision" in self.services else "unavailable",
            "Cloud AI": "connected" if "ai_provider" in self.services else "offline",
        }

        print("\n════════ RENIX CAPABILITIES ════════")
        for name, state in capabilities.items():
            print(f"{name:<20} {state}")
        print("═══════════════════════════════════\n")

    def show_startup_errors(
        self,
    ) -> None:
        """Display startup diagnostics without exposing secrets."""

        print("\n════════ RENIX DIAGNOSTICS ════════")
        if not self.startup_errors:
            print("No startup errors.")
        else:
            for error in self.startup_errors:
                print(f"- {error}")
        print("══════════════════════════════════\n")

    # ========================================================
    # SERVICES
    # ========================================================

    def show_services(
        self,
    ) -> None:

        print(
            "\n════════ RENIX SERVICES ════════"
        )

        if not self.services:

            print(
                "No services initialized."
            )

        else:

            for name, service in (
                self.services.items()
            ):

                print(
                    f"✓ {name} "
                    f"({service.__class__.__name__})"
                )

        print(
            "════════════════════════════════\n"
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    def register_startup_error(
        self,
        component: str,
        error: Exception,
    ) -> None:

        message = (
            f"{component}: {error}"
        )

        self.startup_errors.append(
            message
        )

        logger.warning(
            "%s failed to initialize: %s",
            component,
            error,
        )

    # ========================================================
    # SHUTDOWN
    # ========================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Safely shut down RENIX.
        """

        if not self.running and not self.services:
            return

        logger.info(
            "Shutting down RENIX..."
        )

        self.running = False

        for name, service in list(
            self.services.items()
        ):

            try:

                for method_name in (
                    "shutdown",
                    "stop",
                    "close",
                ):

                    method = getattr(
                        service,
                        method_name,
                        None,
                    )

                    if callable(method):

                        result = method()

                        if asyncio.iscoroutine(
                            result
                        ):
                            await result

                        break

                logger.info(
                    "Stopped service: %s",
                    name,
                )

            except Exception as error:

                logger.error(
                    "Failed to stop %s: %s",
                    name,
                    error,
                )

        self.services.clear()

        logger.info(
            "RENIX shutdown complete."
        )

        print(
            "\nRENIX > Goodbye, Krishna.\n"
        )

    # ========================================================
    # BANNER
    # ========================================================

    @staticmethod
    def print_banner() -> None:

        banner = r"""

██████╗ ███████╗███╗   ██╗██╗██╗  ██╗
██╔══██╗██╔════╝████╗  ██║██║╚██╗██╔╝
██████╔╝█████╗  ██╔██╗ ██║██║ ╚███╔╝
██╔══██╗██╔══╝  ██║╚██╗██║██║ ██╔██╗
██║  ██║███████╗██║ ╚████║██║██╔╝ ██╗
╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝

        PERSONAL ARTIFICIAL INTELLIGENCE
        Version 1.0.0

"""

        print(banner)


# ============================================================
# SIGNAL HANDLING
# ============================================================

def setup_signal_handlers(
    renix: RENIX,
) -> None:
    """
    Register system signal handlers.
    """

    def handle_signal(
        signum: int,
        frame: Any,
    ) -> None:

        logger.info(
            "Shutdown signal received: %s",
            signum,
        )

        renix.running = False

    try:

        signal.signal(
            signal.SIGINT,
            handle_signal,
        )

        signal.signal(
            signal.SIGTERM,
            handle_signal,
        )

    except Exception:

        pass


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

async def main() -> None:
    """
    Main RENIX startup function.
    """

    renix = RENIX()

    setup_signal_handlers(
        renix
    )

    await renix.start()


if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nRENIX terminated."
        )

    except Exception as error:

        logger.exception(
            "Fatal application error: %s",
            error,
        )

        print(
            f"\nFatal Error: {error}"
        )

        sys.exit(1)


