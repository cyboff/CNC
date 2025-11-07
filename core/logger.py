import logging
import os
from logging.handlers import RotatingFileHandler

os.makedirs("data", exist_ok=True)

logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)

# Rotující handler místo FileHandler
fh = RotatingFileHandler(
    "data/app.log",
    maxBytes=2 * 1024 * 1024,  # 2 MB
    backupCount=3,             # uchovat 3 starší logy (app.log.1, app.log.2, app.log.3)
    encoding="utf-8"
)
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)