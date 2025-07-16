import logging
import os

os.makedirs("data", exist_ok=True)

logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler("data/app.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)
