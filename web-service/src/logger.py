import time

class Logger: 
  def __init__(self):
    self.timestamp = time.time()
    self.logFile = f"{self.timestamp}.txt"
    self.logFilePath = f"logs/{self.logFile}"
    self._init()

  
  def _log(self, item):
    with open(self.logFilePath, 'w') as file:
      file.write(f"{self.timestamp} - {item}")

  def _init(self):
    self._log("Service started")

  def log(self, item):
    self._log(item)