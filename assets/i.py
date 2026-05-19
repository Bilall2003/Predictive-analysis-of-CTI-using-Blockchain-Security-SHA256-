import pandas as pd
import numpy as np


list = ['assets/Monday-WorkingHours.csv','assets/Tuesday-WorkingHours.csv','assets/Wednesday-WorkingHours.csv','assets/Thursday-WorkingHours.csv']
for i in list:
    df = pd.read_csv(i)
    print(i,df.shape)
