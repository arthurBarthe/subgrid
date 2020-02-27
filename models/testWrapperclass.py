# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:44:46 2020

@author: Arthur
"""

class Wrapper:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, attr):
        try:
            return getattr(self.wrapped, attr)
        except AttributeError as e:
            raise e

class Wrapped:
    def __init__(self, a):
        self.a = a

    def __repr__(self):
        return str(self.a)

if __name__ == '__main__':
    wrapped = Wrapped(10)
    wrapper = Wrapper(wrapped)
    print(wrapper.a)