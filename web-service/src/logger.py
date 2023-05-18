import time
from datetime import datetime

class Logger: 
  def __init__(self):
    self.timestamp = self._parseTimestamp(time.time())
    self.logFile = f"{self.timestamp}.txt"
    self.logFilePath = f"logs/{self.logFile}"
    self._init()

  def _parseTimestamp(self, timestamp):
    dateTime = datetime.fromtimestamp(timestamp)
    return dateTime.strftime("%Y-%m-%d %H:%M:%S")

  
  def _log(self, item):
    with open(self.logFilePath, 'a') as file:
      file.write(f"{self.timestamp} - {item} \n")

  def _init(self):
    self._log("Service started")

  def log(self, item):
    self._log(item)