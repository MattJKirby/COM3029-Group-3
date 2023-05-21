import os
import time
from datetime import datetime

class Logger: 
  def __init__(self):
    self.timestamp = self._parseTimestamp(time.time())
    self.logFile = f"{self.timestamp}.txt"
    self.logFilePath = f"logs/{self.logFile}"

    if not os.path.isdir("logs"):
      os.mkdir("logs")

    self._init()

  def _parseTimestamp(self, timestamp):
    dateTime = datetime.fromtimestamp(timestamp)
    return dateTime.strftime("%Y-%m-%d %H:%M:%S")

  def _log(self, item):
    with open(self.logFilePath, 'a') as file:
      file.write(f"{self._parseTimestamp(time.time())} - {item} \n")

  def _init(self):
    self._log("Web Service started")

  def log(self, item):
    self._log(item)