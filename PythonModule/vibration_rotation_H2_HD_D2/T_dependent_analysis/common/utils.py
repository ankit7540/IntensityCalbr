# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:31:01 2020

@author: Ankit Raj
"""

# utils.py
from functools import wraps
import gc
import timeit

def MeasureTime(f, no_print=False, disable_gc=False):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        gcold = gc.isenabled()
        if disable_gc:
            gc.disable()
        start_time = timeit.default_timer()
        try:
            result = f(*args, **kwargs)
        finally:
            elapsed = timeit.default_timer() - start_time
            if disable_gc and gcold:
                gc.enable()
            if not no_print:
                print('"{}": {}s'.format(f.__name__, elapsed))
        return result
    return _wrapper

class MeasureBlockTime:
    def __init__(self,name="(block)", no_print=False, disable_gc=False):
        self.name = name
        self.no_print = no_print
        self.disable_gc = disable_gc
    def __enter__(self):
        self.gcold = gc.isenabled()
        if self.disable_gc:
            gc.disable()
        self.start_time = timeit.default_timer()
    def __exit__(self,ty,val,tb):
        self.elapsed = timeit.default_timer() - self.start_time
        if self.disable_gc and self.gcold:
            gc.enable()
        if not self.no_print:
            print('Function "{}": {}s'.format(self.name, self.elapsed))
        return False #re-raise any exceptions