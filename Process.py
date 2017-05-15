#encoding=utf-8
import pandas as pd
from datetime import datetime
import time
import re

x = pd.read_csv('predict.csv',encoding='gbk')
predict = pd.read_csv('predict_set.csv',encoding='gbk')
tutorset = set(x['tutor'])
predict1 = pd.DataFrame()
predict1=predict
for i in range(x.shape[0]):
    if predict.iloc[i]['tutor'] not in tutorset:
        predict1=predict1.drop(i)
predict1.to_csv('graduate/set.csv',index = False, float_format = '%.2f',encoding='gbk') 
 


 

