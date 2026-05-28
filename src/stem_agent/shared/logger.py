import logging
import sys


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_LEVEL = logging.INFO

_configured = False


def configure_logging(level: int = _DEFAULT_LEVEL) -> None:
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure_logging()
    return logging.getLogger(name)