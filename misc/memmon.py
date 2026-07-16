"""
Memory Monitoring Class
=======================

This module presents a single ```MemoryMonitor``` class that allows for memory consumption monitoring
"""

import os
import psutil
from time import sleep

class MemoryMonitor:
    def __init__(self, poll_time=0.05):
        self.keep_measuring = True
        self.peak_rss = 0
        self.process = psutil.Process(os.getpid())
        self.poll_time = poll_time

        self.start_rss = self.process.memory_info().rss   # record the memory already in use before monitoring
        self.peak_rss = self.start_rss

    def measure_usage(self):
        while self.keep_measuring:
            current_rss = self.process.memory_info().rss
            self.peak_rss = max(current_rss, self.peak_rss)

            sleep(self.poll_time)                         # poll time

    def get_consumed_ram(self):
        return self.peak_rss - self.start_rss