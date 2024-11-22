"""Configurations for app logging"""

import logging
import logging.config
import os
from datetime import datetime
from typing import Type
import warnings

LOG_DIR = "logs"

# Generate a unique log filename with timestamp
LOG_FILENAME = os.path.join(
    LOG_DIR, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)

LOG_FORMAT_STANDARD = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_ERROR = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Logging configuration dictionary
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": LOG_FORMAT_STANDARD,
            "datefmt": DATE_FORMAT,
        },
        "error": {
            "format": LOG_FORMAT_ERROR,
            "datefmt": DATE_FORMAT,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "DEBUG",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": LOG_FILENAME,
            "formatter": "error",
            "level": "DEBUG",
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "matplotlib": {
            "handlers": ["console", "file"],
            "level": "WARNING",
            "propagate": False,
        },
        "streamlit": {
            "handlers": ["console", "file"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)


def suppress_matplotlib_warnings() -> None:
    """Additional method to suppress matplotlib warnings

    Returns:
        None:
    """
    # Suppress specific matplotlib warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Set matplotlib logging to critical to minimize output
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.CRITICAL)


def get_logger(name: str) -> Type[logging.Logger]:
    """Get logger for module

    Args:
        name (str): Module name, normally __name__

    Returns:
        Type[logging.Logger]: Module's logger object
    """
    return logging.getLogger(name)


# Suppress warnings on import
suppress_matplotlib_warnings()
