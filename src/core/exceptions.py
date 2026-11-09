"""Core application exceptions."""


class ChurnEngineException(Exception):
    """Base exception for customer churn engine."""
    pass


class ModelNotFoundError(ChurnEngineException):
    """Raised when a required model artifact is not found on disk."""
    pass


class ModelNotLoadedError(ChurnEngineException):
    """Raised when inference is attempted on an uninitialized model manager."""
    pass


class InferenceError(ChurnEngineException):
    """Raised when an error occurs during inference or explainability computation."""
    pass


class DataIngestionError(ChurnEngineException):
    """Raised when dataset fetching or loading fails."""
    pass


class ConfigurationError(ChurnEngineException):
    """Raised when configuration is invalid or missing."""
    pass

