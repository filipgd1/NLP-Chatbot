# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 00:34:48 2020

@author: Gj
"""

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)