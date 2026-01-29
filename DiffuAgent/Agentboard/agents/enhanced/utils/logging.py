"""
Enhanced logging utilities for AgentBoard enhanced modules.

This module provides a unified logging interface that integrates with
AgentBoard's existing logging system while providing enhanced modules
with their own logging namespace.
"""
import logging
from utils.logging.agent_logger import AgentLogger


def get_logger(name, filepath=None):
    """
    Get a logger instance for enhanced modules.

    This creates a logger with the enhanced module's namespace, integrating
    seamlessly with AgentBoard's existing colored logging system.

    Args:
        name: Logger name (typically __name__ of the module)
        filepath: Optional file path for log output

    Returns:
        AgentLogger instance configured for the enhanced module

    Example:
        from agents.enhanced.utils.logging import get_logger
        logger = get_logger(__name__)
        logger.info("DynamicMemory initialized")
        logger.debug("Detailed debug info")
    """
    logger = AgentLogger(name, filepath=filepath)

    # Set to INFO level by default for enhanced modules
    # Can be configured to DEBUG for more verbose output
    logger.setLevel(logging.INFO)

    return logger


# Log levels for enhanced modules
# Use these for consistent logging across enhanced modules
class EnhancedLogLevels:
    """Standardized log levels for enhanced modules."""

    @staticmethod
    def debug(logger, message):
        """Debug-level logging for detailed diagnostics."""
        logger.debug(message)

    @staticmethod
    def info(logger, message):
        """Info-level logging for general information."""
        logger.info(message)

    @staticmethod
    def warning(logger, message):
        """Warning-level logging for potential issues."""
        logger.warning(message)

    @staticmethod
    def error(logger, message):
        """Error-level logging for errors and exceptions."""
        logger.error(message)

    @staticmethod
    def success(logger, message):
        """Log successful operations."""
        logger.info(f"✓ {message}")

    @staticmethod
    def failure(logger, message):
        """Log failed operations."""
        logger.warning(f"✗ {message}")

    @staticmethod
    def init(logger, component, details=""):
        """Log component initialization."""
        if details:
            logger.info(f"[INIT] {component}: {details}")
        else:
            logger.info(f"[INIT] {component}")

    @staticmethod
    def step(logger, step_num, details=""):
        """Log step/progress information."""
        if details:
            logger.info(f"[STEP {step_num}] {details}")
        else:
            logger.info(f"[STEP {step_num}]")
