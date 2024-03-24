import logging
import sys

def pytest_configure(config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/tests.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        encoding='utf-8'
    )
    logging.info("Pytest configure...")
