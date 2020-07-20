# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:11:59 2020

@author: Gj
"""

import pandas as pd

path_to_excel = 'dataset_brainster.xlsx'
data_excel = pd.read_excel(path_to_excel, encoding='utf-8')

for col in data_excel.columns:
    data_excel[col].astype(str)

data_excel.to_csv (r'data_brainster.csv', index = None, 
                   header=True, encoding='utf-8-sig')