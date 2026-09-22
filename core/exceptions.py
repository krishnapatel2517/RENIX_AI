"""
RENIX AI
Core Exceptions

Centralized exception hierarchy for the RENIX system.

All RENIX-specific exceptions should inherit from
RENIXError so that the orchestrator and other modules
can catch RENIX failures consistently.
"""

from __future__ import annotations

from typing import Any, Optional


# ============================================================================
# BASE EXCEPTION
# ============================================================================

class RENIXError(Exception):
    """
    Base exception for all RENIX-specific errors.
    """

    code: str = "RENIX_ERROR"

    def __init__(
        self,
        message: str = "An unknown RENIX error occurred.",
        *,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:

        self.message = str(message)

        self.details = dict(
            details or {}
        )

        self.cause = cause

        super().__init__(
            self.message
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert the exception into a serializable dictionary.
        """

        return {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
            "cause": (
                str(self.cause)
                if self.cause is not None
                else None
            ),
        }

    def __str__(
        self,
    ) -> str:

        return self.message


# ============================================================================
# CONFIGURATION ERRORS
# ============================================================================

class ConfigurationError(RENIXError):
    """
    Raised when RENIX configuration is invalid.
    """

    code = "CONFIGURATION_ERROR"


class MissingConfigurationError(ConfigurationError):
    """
    Raised when required configuration is missing.
    """

    code = "MISSING_CONFIGURATION"


class InvalidConfigurationError(ConfigurationError):
    """
    Raised when configuration contains invalid values.
    """

    code = "INVALID_CONFIGURATION"


# ============================================================================
# INITIALIZATION ERRORS
# ============================================================================

class InitializationError(RENIXError):
    """
    Raised when a RENIX component cannot initialize.
    """

    code = "INITIALIZATION_ERROR"


class StartupError(InitializationError):
    """
    Raised when RENIX cannot start correctly.
    """

    code = "STARTUP_ERROR"


class ShutdownError(RENIXError):
    """
    Raised when RENIX cannot shut down correctly.
    """

    code = "SHUTDOWN_ERROR"


# ============================================================================
# SERVICE ERRORS
# ============================================================================

class ServiceError(RENIXError):
    """
    Base exception for service-related failures.
    """

    code = "SERVICE_ERROR"


class ServiceNotFoundError(ServiceError):
    """
    Raised when a requested service does not exist.
    """

    code = "SERVICE_NOT_FOUND"


class ServiceAlreadyRegisteredError(ServiceError):
    """
    Raised when a service is registered twice.
    """

    code = "SERVICE_ALREADY_REGISTERED"


class ServiceUnavailableError(ServiceError):
    """
    Raised when a service is temporarily unavailable.
    """

    code = "SERVICE_UNAVAILABLE"


class ServiceInitializationError(ServiceError):
    """
    Raised when a service fails during initialization.
    """

    code = "SERVICE_INITIALIZATION_ERROR"


# ============================================================================
# PLUGIN ERRORS
# ============================================================================

class PluginError(RENIXError):
    """
    Base exception for plugin-related failures.
    """

    code = "PLUGIN_ERROR"


class PluginNotFoundError(PluginError):
    """
    Raised when a plugin cannot be found.
    """

    code = "PLUGIN_NOT_FOUND"


class PluginLoadError(PluginError):
    """
    Raised when a plugin cannot be loaded.
    """

    code = "PLUGIN_LOAD_ERROR"


class PluginInitializationError(PluginError):
    """
    Raised when a plugin cannot initialize.
    """

    code = "PLUGIN_INITIALIZATION_ERROR"


class PluginExecutionError(PluginError):
    """
    Raised when a plugin fails during execution.
    """

    code = "PLUGIN_EXECUTION_ERROR"


# ============================================================================
# CAPABILITY ERRORS
# ============================================================================

class CapabilityError(RENIXError):
    """
    Base exception for capability-related failures.
    """

    code = "CAPABILITY_ERROR"


class CapabilityNotFoundError(CapabilityError):
    """
    Raised when a requested capability does not exist.
    """

    code = "CAPABILITY_NOT_FOUND"


class CapabilityUnavailableError(CapabilityError):
    """
    Raised when a capability is currently unavailable.
    """

    code = "CAPABILITY_UNAVAILABLE"


class CapabilityPermissionError(CapabilityError):
    """
    Raised when a capability cannot be used because
    the required permission is unavailable.
    """

    code = "CAPABILITY_PERMISSION_ERROR"


# ============================================================================
# COMMAND ERRORS
# ============================================================================

class CommandError(RENIXError):
    """
    Base exception for command-related failures.
    """

    code = "COMMAND_ERROR"


class InvalidCommandError(CommandError):
    """
    Raised when a command is malformed or invalid.
    """

    code = "INVALID_COMMAND"


class CommandNotFoundError(CommandError):
    """
    Raised when a command cannot be resolved.
    """

    code = "COMMAND_NOT_FOUND"


class CommandExecutionError(CommandError):
    """
    Raised when command execution fails.
    """

    code = "COMMAND_EXECUTION_ERROR"


class CommandTimeoutError(CommandError):
    """
    Raised when command execution exceeds its timeout.
    """

    code = "COMMAND_TIMEOUT"


# ============================================================================
# TASK ERRORS
# ============================================================================

class TaskError(RENIXError):
    """
    Base exception for task-related failures.
    """

    code = "TASK_ERROR"


class TaskNotFoundError(TaskError):
    """
    Raised when a task cannot be found.
    """

    code = "TASK_NOT_FOUND"


class TaskAlreadyExistsError(TaskError):
    """
    Raised when a duplicate task is created.
    """

    code = "TASK_ALREADY_EXISTS"


class TaskExecutionError(TaskError):
    """
    Raised when a task fails during execution.
    """

    code = "TASK_EXECUTION_ERROR"


class TaskCancelledError(TaskError):
    """
    Raised when a task is cancelled.
    """

    code = "TASK_CANCELLED"


class TaskTimeoutError(TaskError):
    """
    Raised when a task exceeds its allowed execution time.
    """

    code = "TASK_TIMEOUT"


# ============================================================================
# PLANNING / REASONING ERRORS
# ============================================================================

class PlanningError(RENIXError):
    """
    Base exception for planning failures.
    """

    code = "PLANNING_ERROR"


class PlanningFailedError(PlanningError):
    """
    Raised when RENIX cannot create a valid plan.
    """

    code = "PLANNING_FAILED"


class ReasoningError(RENIXError):
    """
    Base exception for reasoning failures.
    """

    code = "REASONING_ERROR"


class ReasoningFailedError(ReasoningError):
    """
    Raised when reasoning fails.
    """

    code = "REASONING_FAILED"


class DecisionError(RENIXError):
    """
    Base exception for decision-making failures.
    """

    code = "DECISION_ERROR"


class DecisionFailedError(DecisionError):
    """
    Raised when RENIX cannot make a valid decision.
    """

    code = "DECISION_FAILED"


# ============================================================================
# AI / MODEL ERRORS
# ============================================================================

class AIError(RENIXError):
    """
    Base exception for AI-related failures.
    """

    code = "AI_ERROR"


class ModelError(AIError):
    """
    Base exception for model failures.
    """

    code = "MODEL_ERROR"


class ModelNotFoundError(ModelError):
    """
    Raised when a requested model cannot be found.
    """

    code = "MODEL_NOT_FOUND"


class ModelUnavailableError(ModelError):
    """
    Raised when a model is temporarily unavailable.
    """

    code = "MODEL_UNAVAILABLE"


class ModelAuthenticationError(ModelError):
    """
    Raised when model authentication fails.
    """

    code = "MODEL_AUTHENTICATION_ERROR"


class ModelRateLimitError(ModelError):
    """
    Raised when an AI provider rate limit is reached.
    """

    code = "MODEL_RATE_LIMIT"


class ModelTimeoutError(ModelError):
    """
    Raised when an AI model request times out.
    """

    code = "MODEL_TIMEOUT"


class ModelResponseError(ModelError):
    """
    Raised when an AI provider returns an invalid response.
    """

    code = "MODEL_RESPONSE_ERROR"


# ============================================================================
# MEMORY ERRORS
# ============================================================================

class MemoryError(RENIXError):
    """
    Base exception for memory-related failures.
    """

    code = "MEMORY_ERROR"


class MemoryNotFoundError(MemoryError):
    """
    Raised when requested memory cannot be found.
    """

    code = "MEMORY_NOT_FOUND"


class MemoryStorageError(MemoryError):
    """
    Raised when memory cannot be stored.
    """

    code = "MEMORY_STORAGE_ERROR"


class MemoryRetrievalError(MemoryError):
    """
    Raised when memory retrieval fails.
    """

    code = "MEMORY_RETRIEVAL_ERROR"


class MemorySerializationError(MemoryError):
    """
    Raised when memory cannot be serialized.
    """

    code = "MEMORY_SERIALIZATION_ERROR"


class MemoryPermissionError(MemoryError):
    """
    Raised when memory access is not permitted.
    """

    code = "MEMORY_PERMISSION_ERROR"


# ============================================================================
# VOICE ERRORS
# ============================================================================

class VoiceError(RENIXError):
    """
    Base exception for voice-related failures.
    """

    code = "VOICE_ERROR"


class MicrophoneError(VoiceError):
    """
    Raised when microphone access fails.
    """

    code = "MICROPHONE_ERROR"


class SpeechRecognitionError(VoiceError):
    """
    Raised when speech recognition fails.
    """

    code = "SPEECH_RECOGNITION_ERROR"


class SpeechToTextError(VoiceError):
    """
    Raised when speech-to-text processing fails.
    """

    code = "SPEECH_TO_TEXT_ERROR"


class TextToSpeechError(VoiceError):
    """
    Raised when text-to-speech processing fails.
    """

    code = "TEXT_TO_SPEECH_ERROR"


class WakeWordError(VoiceError):
    """
    Raised when wake-word detection fails.
    """

    code = "WAKE_WORD_ERROR"


class AudioProcessingError(VoiceError):
    """
    Raised when audio processing fails.
    """

    code = "AUDIO_PROCESSING_ERROR"


# ============================================================================
# VISION ERRORS
# ============================================================================

class VisionError(RENIXError):
    """
    Base exception for vision-related failures.
    """

    code = "VISION_ERROR"


class CameraError(VisionError):
    """
    Raised when camera access fails.
    """

    code = "CAMERA_ERROR"


class FaceDetectionError(VisionError):
    """
    Raised when face detection fails.
    """

    code = "FACE_DETECTION_ERROR"


class FaceRecognitionError(VisionError):
    """
    Raised when face recognition fails.
    """

    code = "FACE_RECOGNITION_ERROR"


class ObjectDetectionError(VisionError):
    """
    Raised when object detection fails.
    """

    code = "OBJECT_DETECTION_ERROR"


class OCRProcessingError(VisionError):
    """
    Raised when OCR processing fails.
    """

    code = "OCR_PROCESSING_ERROR"


# ============================================================================
# GESTURE ERRORS
# ============================================================================

class GestureError(RENIXError):
    """
    Base exception for gesture-related failures.
    """

    code = "GESTURE_ERROR"


class GestureDetectionError(GestureError):
    """
    Raised when gesture detection fails.
    """

    code = "GESTURE_DETECTION_ERROR"


class GestureClassificationError(GestureError):
    """
    Raised when gesture classification fails.
    """

    code = "GESTURE_CLASSIFICATION_ERROR"


class GestureMappingError(GestureError):
    """
    Raised when a gesture cannot be mapped to an action.
    """

    code = "GESTURE_MAPPING_ERROR"


# ============================================================================
# COMPUTER CONTROL ERRORS
# ============================================================================

class ComputerError(RENIXError):
    """
    Base exception for computer-control failures.
    """

    code = "COMPUTER_ERROR"


class MouseError(ComputerError):
    """
    Raised when mouse control fails.
    """

    code = "MOUSE_ERROR"


class KeyboardError(ComputerError):
    """
    Raised when keyboard control fails.
    """

    code = "KEYBOARD_ERROR"


class ScreenError(ComputerError):
    """
    Raised when screen operations fail.
    """

    code = "SCREEN_ERROR"


class ApplicationError(ComputerError):
    """
    Raised when application control fails.
    """

    code = "APPLICATION_ERROR"


class ClipboardError(ComputerError):
    """
    Raised when clipboard operations fail.
    """

    code = "CLIPBOARD_ERROR"


# ============================================================================
# FILE ERRORS
# ============================================================================

class FileSystemError(RENIXError):
    """
    Base exception for filesystem-related failures.
    """

    code = "FILESYSTEM_ERROR"


class FileNotFoundError(FileSystemError):
    """
    Raised when a requested file does not exist.
    """

    code = "FILE_NOT_FOUND"


class FileAccessError(FileSystemError):
    """
    Raised when a file cannot be accessed.
    """

    code = "FILE_ACCESS_ERROR"


class FileOperationError(FileSystemError):
    """
    Raised when a file operation fails.
    """

    code = "FILE_OPERATION_ERROR"


class FilePermissionError(FileSystemError):
    """
    Raised when filesystem permissions prevent an operation.
    """

    code = "FILE_PERMISSION_ERROR"


class FileAlreadyExistsError(FileSystemError):
    """
    Raised when an operation would overwrite an existing file
    without permission.
    """

    code = "FILE_ALREADY_EXISTS"


# ============================================================================
# BROWSER ERRORS
# ============================================================================

class BrowserError(RENIXError):
    """
    Base exception for browser-related failures.
    """

    code = "BROWSER_ERROR"


class BrowserNotAvailableError(BrowserError):
    """
    Raised when a browser cannot be accessed.
    """

    code = "BROWSER_NOT_AVAILABLE"


class WebPageError(BrowserError):
    """
    Raised when a webpage cannot be loaded or processed.
    """

    code = "WEBPAGE_ERROR"


class WebSearchError(BrowserError):
    """
    Raised when web search fails.
    """

    code = "WEB_SEARCH_ERROR"


class DownloadError(BrowserError):
    """
    Raised when a browser download fails.
    """

    code = "DOWNLOAD_ERROR"


# ============================================================================
# CODING ERRORS
# ============================================================================

class CodingError(RENIXError):
    """
    Base exception for coding-related failures.
    """

    code = "CODING_ERROR"


class CodeExecutionError(CodingError):
    """
    Raised when generated or existing code fails to execute.
    """

    code = "CODE_EXECUTION_ERROR"


class CodeAnalysisError(CodingError):
    """
    Raised when code analysis fails.
    """

    code = "CODE_ANALYSIS_ERROR"


class BuildError(CodingError):
    """
    Raised when a project build fails.
    """

    code = "BUILD_ERROR"


class TestError(CodingError):
    """
    Raised when automated tests fail to execute.
    """

    code = "TEST_ERROR"


class DependencyError(CodingError):
    """
    Raised when dependency operations fail.
    """

    code = "DEPENDENCY_ERROR"


class GitError(CodingError):
    """
    Raised when Git operations fail.
    """

    code = "GIT_ERROR"


# ============================================================================
# AUTOMATION ERRORS
# ============================================================================

class AutomationError(RENIXError):
    """
    Base exception for automation failures.
    """

    code = "AUTOMATION_ERROR"


class WorkflowError(AutomationError):
    """
    Raised when a workflow fails.
    """

    code = "WORKFLOW_ERROR"


class TriggerError(AutomationError):
    """
    Raised when an automation trigger fails.
    """

    code = "TRIGGER_ERROR"


class MacroError(AutomationError):
    """
    Raised when a macro fails.
    """

    code = "MACRO_ERROR"


class AutomationTimeoutError(AutomationError):
    """
    Raised when automation exceeds its timeout.
    """

    code = "AUTOMATION_TIMEOUT"


# ============================================================================
# SECURITY ERRORS
# ============================================================================

class SecurityError(RENIXError):
    """
    Base exception for security failures.
    """

    code = "SECURITY_ERROR"


class AuthenticationError(SecurityError):
    """
    Raised when authentication fails.
    """

    code = "AUTHENTICATION_ERROR"


class AuthorizationError(SecurityError):
    """
    Raised when an operation is not authorized.
    """

    code = "AUTHORIZATION_ERROR"


class PermissionDeniedError(SecurityError):
    """
    Raised when permission is explicitly denied.
    """

    code = "PERMISSION_DENIED"


class SecurityPolicyError(SecurityError):
    """
    Raised when an operation violates a security policy.
    """

    code = "SECURITY_POLICY_ERROR"


class ConfirmationRequiredError(SecurityError):
    """
    Raised when an operation requires user confirmation.
    """

    code = "CONFIRMATION_REQUIRED"


class SecureModeError(SecurityError):
    """
    Raised when secure mode blocks an operation.
    """

    code = "SECURE_MODE_ERROR"


# ============================================================================
# DEVICE ERRORS
# ============================================================================

class DeviceError(RENIXError):
    """
    Base exception for device-related failures.
    """

    code = "DEVICE_ERROR"


class DeviceNotFoundError(DeviceError):
    """
    Raised when a device cannot be found.
    """

    code = "DEVICE_NOT_FOUND"


class DeviceConnectionError(DeviceError):
    """
    Raised when a device connection fails.
    """

    code = "DEVICE_CONNECTION_ERROR"


class DevicePairingError(DeviceError):
    """
    Raised when device pairing fails.
    """

    code = "DEVICE_PAIRING_ERROR"


class DeviceCommunicationError(DeviceError):
    """
    Raised when communication with a device fails.
    """

    code = "DEVICE_COMMUNICATION_ERROR"


# ============================================================================
# ROBOTICS ERRORS
# ============================================================================

class RoboticsError(RENIXError):
    """
    Base exception for robotics failures.
    """

    code = "ROBOTICS_ERROR"


class RobotConnectionError(RoboticsError):
    """
    Raised when RENIX cannot connect to a robot.
    """

    code = "ROBOT_CONNECTION_ERROR"


class RobotControlError(RoboticsError):
    """
    Raised when robot control fails.
    """

    code = "ROBOT_CONTROL_ERROR"


class RobotSafetyError(RoboticsError):
    """
    Raised when a robotics safety condition is violated.
    """

    code = "ROBOT_SAFETY_ERROR"


# ============================================================================
# DATABASE ERRORS
# ============================================================================

class DatabaseError(RENIXError):
    """
    Base exception for database failures.
    """

    code = "DATABASE_ERROR"


class DatabaseConnectionError(DatabaseError):
    """
    Raised when the database cannot be opened.
    """

    code = "DATABASE_CONNECTION_ERROR"


class DatabaseQueryError(DatabaseError):
    """
    Raised when a database query fails.
    """

    code = "DATABASE_QUERY_ERROR"


class DatabaseMigrationError(DatabaseError):
    """
    Raised when a database migration fails.
    """

    code = "DATABASE_MIGRATION_ERROR"


# ============================================================================
# INTEGRATION ERRORS
# ============================================================================

class IntegrationError(RENIXError):
    """
    Base exception for external integration failures.
    """

    code = "INTEGRATION_ERROR"


class ProviderError(IntegrationError):
    """
    Raised when an external provider fails.
    """

    code = "PROVIDER_ERROR"


class APIError(IntegrationError):
    """
    Raised when an external API request fails.
    """

    code = "API_ERROR"


class APIAuthenticationError(APIError):
    """
    Raised when API authentication fails.
    """

    code = "API_AUTHENTICATION_ERROR"


class APIRateLimitError(APIError):
    """
    Raised when an API rate limit is reached.
    """

    code = "API_RATE_LIMIT"


class APIResponseError(APIError):
    """
    Raised when an API response is invalid.
    """

    code = "API_RESPONSE_ERROR"


# ============================================================================
# SCHEDULER ERRORS
# ============================================================================

class SchedulerError(RENIXError):
    """
    Base exception for scheduler failures.
    """

    code = "SCHEDULER_ERROR"


class ScheduleNotFoundError(SchedulerError):
    """
    Raised when a scheduled task cannot be found.
    """

    code = "SCHEDULE_NOT_FOUND"


class ScheduleConflictError(SchedulerError):
    """
    Raised when schedules conflict.
    """

    code = "SCHEDULE_CONFLICT"


class InvalidScheduleError(SchedulerError):
    """
    Raised when a schedule is invalid.
    """

    code = "INVALID_SCHEDULE"


# ============================================================================
# MEMORY / STATE / SESSION ERRORS
# ============================================================================

class StateError(RENIXError):
    """
    Base exception for state-management failures.
    """

    code = "STATE_ERROR"


class InvalidStateError(StateError):
    """
    Raised when RENIX enters or receives an invalid state.
    """

    code = "INVALID_STATE"


class SessionError(RENIXError):
    """
    Base exception for session-management failures.
    """

    code = "SESSION_ERROR"


class SessionNotFoundError(SessionError):
    """
    Raised when a session cannot be found.
    """

    code = "SESSION_NOT_FOUND"


class SessionExpiredError(SessionError):
    """
    Raised when a session has expired.
    """

    code = "SESSION_EXPIRED"


# ============================================================================
# NETWORK ERRORS
# ============================================================================

class NetworkError(RENIXError):
    """
    Base exception for network failures.
    """

    code = "NETWORK_ERROR"


class NetworkConnectionError(NetworkError):
    """
    Raised when a network connection cannot be established.
    """

    code = "NETWORK_CONNECTION_ERROR"


class NetworkTimeoutError(NetworkError):
    """
    Raised when a network operation times out.
    """

    code = "NETWORK_TIMEOUT"


# ============================================================================
# HEALTH ERRORS
# ============================================================================

class HealthError(RENIXError):
    """
    Base exception for health-monitoring failures.
    """

    code = "HEALTH_ERROR"


class HealthCheckError(HealthError):
    """
    Raised when a health check fails unexpectedly.
    """

    code = "HEALTH_CHECK_ERROR"


class ComponentUnhealthyError(HealthError):
    """
    Raised when a critical component becomes unhealthy.
    """

    code = "COMPONENT_UNHEALTHY"


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(RENIXError):
    """
    Base exception for validation failures.
    """

    code = "VALIDATION_ERROR"


class InvalidInputError(ValidationError):
    """
    Raised when user or system input is invalid.
    """

    code = "INVALID_INPUT"


class MissingInputError(ValidationError):
    """
    Raised when required input is missing.
    """

    code = "MISSING_INPUT"


# ============================================================================
# RESOURCE ERRORS
# ============================================================================

class ResourceError(RENIXError):
    """
    Base exception for resource-related failures.
    """

    code = "RESOURCE_ERROR"


class ResourceNotFoundError(ResourceError):
    """
    Raised when a required resource cannot be found.
    """

    code = "RESOURCE_NOT_FOUND"


class ResourceUnavailableError(ResourceError):
    """
    Raised when a resource is unavailable.
    """

    code = "RESOURCE_UNAVAILABLE"


class ResourceLimitError(ResourceError):
    """
    Raised when a resource limit is exceeded.
    """

    code = "RESOURCE_LIMIT"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "RENIXError",

    "ConfigurationError",
    "MissingConfigurationError",
    "InvalidConfigurationError",

    "InitializationError",
    "StartupError",
    "ShutdownError",

    "ServiceError",
    "ServiceNotFoundError",
    "ServiceAlreadyRegisteredError",
    "ServiceUnavailableError",
    "ServiceInitializationError",

    "PluginError",
    "PluginNotFoundError",
    "PluginLoadError",
    "PluginInitializationError",
    "PluginExecutionError",

    "CapabilityError",
    "CapabilityNotFoundError",
    "CapabilityUnavailableError",
    "CapabilityPermissionError",

    "CommandError",
    "InvalidCommandError",
    "CommandNotFoundError",
    "CommandExecutionError",
    "CommandTimeoutError",

    "TaskError",
    "TaskNotFoundError",
    "TaskAlreadyExistsError",
    "TaskExecutionError",
    "TaskCancelledError",
    "TaskTimeoutError",

    "PlanningError",
    "PlanningFailedError",
    "ReasoningError",
    "ReasoningFailedError",
    "DecisionError",
    "DecisionFailedError",

    "AIError",
    "ModelError",
    "ModelNotFoundError",
    "ModelUnavailableError",
    "ModelAuthenticationError",
    "ModelRateLimitError",
    "ModelTimeoutError",
    "ModelResponseError",

    "MemoryError",
    "MemoryNotFoundError",
    "MemoryStorageError",
    "MemoryRetrievalError",
    "MemorySerializationError",
    "MemoryPermissionError",

    "VoiceError",
    "MicrophoneError",
    "SpeechRecognitionError",
    "SpeechToTextError",
    "TextToSpeechError",
    "WakeWordError",
    "AudioProcessingError",

    "VisionError",
    "CameraError",
    "FaceDetectionError",
    "FaceRecognitionError",
    "ObjectDetectionError",
    "OCRProcessingError",

    "GestureError",
    "GestureDetectionError",
    "GestureClassificationError",
    "GestureMappingError",

    "ComputerError",
    "MouseError",
    "KeyboardError",
    "ScreenError",
    "ApplicationError",
    "ClipboardError",

    "FileSystemError",
    "FileNotFoundError",
    "FileAccessError",
    "FileOperationError",
    "FilePermissionError",
    "FileAlreadyExistsError",

    "BrowserError",
    "BrowserNotAvailableError",
    "WebPageError",
    "WebSearchError",
    "DownloadError",

    "CodingError",
    "CodeExecutionError",
    "CodeAnalysisError",
    "BuildError",
    "TestError",
    "DependencyError",
    "GitError",

    "AutomationError",
    "WorkflowError",
    "TriggerError",
    "MacroError",
    "AutomationTimeoutError",

    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    "PermissionDeniedError",
    "SecurityPolicyError",
    "ConfirmationRequiredError",
    "SecureModeError",

    "DeviceError",
    "DeviceNotFoundError",
    "DeviceConnectionError",
    "DevicePairingError",
    "DeviceCommunicationError",

    "RoboticsError",
    "RobotConnectionError",
    "RobotControlError",
    "RobotSafetyError",

    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseMigrationError",

    "IntegrationError",
    "ProviderError",
    "APIError",
    "APIAuthenticationError",
    "APIRateLimitError",
    "APIResponseError",

    "SchedulerError",
    "ScheduleNotFoundError",
    "ScheduleConflictError",
    "InvalidScheduleError",

    "StateError",
    "InvalidStateError",

    "SessionError",
    "SessionNotFoundError",
    "SessionExpiredError",

    "NetworkError",
    "NetworkConnectionError",
    "NetworkTimeoutError",

    "HealthError",
    "HealthCheckError",
    "ComponentUnhealthyError",

    "ValidationError",
    "InvalidInputError",
    "MissingInputError",

    "ResourceError",
    "ResourceNotFoundError",
    "ResourceUnavailableError",
    "ResourceLimitError",
]


