# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:23:04 2023

@author: Renea

Calls to the Polygon API using their Free service has a limit of 5 per minute. 
The below allows us to impose a delay so we do not exceed the limit

"""
import time

class delay:
    def __init__(self):
        self.now = time.time() - 60
        self.history=[]
        for _ in range(5):
            self.history.append(self.now)
            
    def step(self):
        
        self.timetrack()
        
        if (self.check_history()):
            print(f"Exceeded 5 queries per minute - pausing {int(self.get_wait_time())} seconds")
            time.sleep(self.get_wait_time() + 0.5)
    
    def timetrack(self):
        self.now = time.time()
        self.history.append(self.now)
    
    def time_elapsed(self) -> float:
        return (time.time() - self.now)
    
    def history_time_elapsed(self) -> list[float]:
        ''' returns a list containing the seconds elapsed over the past 5 steps'''
        return [time.time() - x for x in self.history]

    def check_history(self) -> bool:
        ''' returns true if the 5th last query occurred less than 60 seconds ago'''
        history = self.history_time_elapsed()
        if history[-5] < 60:
            return True
        return False
    
    def get_wait_time(self) -> float:
        ''' returns 60 minus the time passed since the 5th last query '''
        wait = self.history_time_elapsed()
        return (60 - wait[-5])
        
