# %%
from typing import Optional, Literal
from logging import (
    getLogger,
    Formatter,
    FileHandler,
    StreamHandler
)

LOGGER_LEVELS = Literal[
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL"
]

# %%
def get_logger(
        level: LOGGER_LEVELS = "INFO",
        file_name: Optional[str]=None,
        using_file_handler: bool=True,
        using_stream_handler: bool=True
    ):
    # root logger settings
    logger = getLogger("robat")
    logger.propagate = False # to prevent double logging on stream handler
    logger.setLevel(level)

    if logger.hasHandlers():
        # Clear existing handlers to avoid duplicate logs
        logger.handlers.clear()

    formatter = Formatter(
        "[%(levelname)s] [%(asctime)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if using_file_handler and file_name is None:
        raise ValueError("file_name must be provided if using_file_handler is True")
    if using_file_handler:
        file_handler = FileHandler(file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if using_stream_handler:
        stream_handler = StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger