import logging
import sys
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Prevent duplicate handlers on reload
        logger.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)

        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a configured logger."""
    return setup_logger(name)
