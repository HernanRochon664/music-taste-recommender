"""
Logging configuration for the project

Provides centralized logging setup with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import io


class UTF8StreamHandler(logging.StreamHandler):
    """
    Custom StreamHandler that forces UTF-8 encoding

    Fixes Windows console emoji/unicode issues
    """
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout

        # Force UTF-8 encoding
        if hasattr(stream, 'buffer'):
            stream = io.TextIOWrapper(
                stream.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )

        super().__init__(stream)

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Encode to UTF-8 and decode, replacing errors
            stream.write(msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers

    Args:
        name: Logger name (usually __name__ of the module)
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # File handler
    if log_to_file:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_dir_path / f"{name.replace('.', '_')}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Console handler with UTF-8 support
    if log_to_console:
        console_handler = UTF8StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with default configuration

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    # Try to load config from YAML
    try:
        from config import config
        log_level = config.get('logging.level', 'INFO')
        log_dir = config.get('paths.logs', 'logs')
        file_enabled = config.get('logging.file_enabled', True)
        console_enabled = config.get('logging.console_enabled', True)
    except:
        # Fallback to defaults if config not available
        log_level = 'INFO'
        log_dir = 'logs'
        file_enabled = True
        console_enabled = True

    return setup_logger(
        name=name,
        log_dir=log_dir,
        level=log_level,
        log_to_file=file_enabled,
        log_to_console=console_enabled
    )