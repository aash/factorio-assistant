import logging
import sys

def pytest_configure(config):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    

def pytest_sessionstart(session):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/tests.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        encoding='utf-8'
    )
    logging.info("Test session is starting.")
