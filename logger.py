from datetime import datetime
import logging
import sys
import os

l = logging.getLogger("metrics")
l.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
filename = input("Log file name ") or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
log_filename = f"logs/{filename}.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
fileHandler = logging.FileHandler(log_filename, mode="w", encoding=None, delay=False)
fileHandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s; %(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)
l.addHandler(fileHandler)
l.addHandler(consoleHandler)
