from datetime import datetime
import logging
import sys

l = logging.getLogger("metrics")
l.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler(f"logs/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log",mode='a')
fileHandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s::%(name)s:: %(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)
l.addHandler(fileHandler)
l.addHandler(consoleHandler)
