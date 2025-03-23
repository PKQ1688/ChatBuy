import logging
import os

from rich.console import Console
from rich.logging import RichHandler

# Define module name and create log directory
MODULE_NAME = "chatbuy"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure Rich console for stdout
console = Console()

# Configure root logger with Rich handler for stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            enable_link_path=False,
        )
    ],
)

# Create module logger
logger = logging.getLogger(MODULE_NAME)
logger.setLevel(logging.DEBUG)

# Prevent propagation to root logger to avoid duplicate logs
logger.propagate = False

# Remove any existing handlers to avoid duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add Rich handler for console output
logger.addHandler(
    RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_path=False,
    )
)

# Configure file handler for logging
file_handler = logging.FileHandler(f"{LOG_DIR}/chatbuy.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

file_handler.setFormatter(
    logging.Formatter(
        f"%(asctime)s | %(levelname)-8s | {MODULE_NAME} | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logger.addHandler(file_handler)


# Define helper methods to maintain similar API to loguru
def get_logger():
    """Return the configured logger instance."""
    return logger


# Add context information wrapper (simpler than loguru's bind)
class ContextLogger:
    """A wrapper class that adds module context to logging messages.
    
    This provides a simpler alternative to loguru's bind functionality.
    """
    def __init__(self, logger, module):
        self.logger = logger
        self.module = module

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(f"[{self.module}] {msg}", *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(f"[{self.module}] {msg}", *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(f"[{self.module}] {msg}", *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(f"[{self.module}] {msg}", *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(f"[{self.module}] {msg}", *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.logger.exception(f"[{self.module}] {msg}", *args, **kwargs)


# Create a contextualized logger for this module
logger = ContextLogger(logger, MODULE_NAME)
