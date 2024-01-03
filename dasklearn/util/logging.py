import logging
import os


def setup_logging(data_dir: str, log_file_name: str, log_level: str = "INFO") -> None:
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    logger.handlers.clear()

    # Create handlers (file and console)
    file_handler = logging.FileHandler(os.path.join(data_dir, log_file_name), mode='w')
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setFormatter(logging.Formatter(log_format))

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)