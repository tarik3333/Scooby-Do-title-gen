# -*- coding: utf-8 -*-
"""
Created on Wed May  16 22:22:08 2021

@author: Tarık Buğra Tufan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Scooby-Doo Completed.csv")
#print(df["title"].values)
with open('Scooby_titles.txt', 'w') as f:
    titles=[i for i in df["title"].values]
    titles=str(titles)
    f.write(titles)
