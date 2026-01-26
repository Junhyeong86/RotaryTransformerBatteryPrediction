# %%
from typing import (
    Optional,
    Literal,
    Dict,
    Any
)

LOGGER_LEVELS = Literal[
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL"
]

def generate_logging_config(
        logger_name: str = "performance_reasoning",
        level: LOGGER_LEVELS = "INFO",
        using_stream_handler: bool = True,
        using_file_handler: bool = True,
        file_name: Optional[str] = None
    ) -> Dict[str, Any]:
    if using_file_handler and file_name is None:
        raise ValueError("file_name must be provided if using_file_handler is True")

    handlers = {}
    handler_names = []
    if using_stream_handler:
        handlers["stream_handler"] = {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        }
        handler_names.append("stream_handler")

    if using_file_handler:
        handlers["file_handler"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": file_name,
            "mode": "a",
            "encoding": "utf-8"
        }
        handler_names.append("file_handler")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(levelname)s] [%(asctime)s] [%(name)s]: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": handlers,
        "loggers": {
            logger_name: {
                "level": level,
                "handlers": handler_names,
                "propagate": False
            }
        }
    }
    return config