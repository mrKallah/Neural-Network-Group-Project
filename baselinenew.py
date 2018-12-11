# -*- coding: utf-8 -*-
'@author: Zhengkun Yao'

import pandas as pd
import numpy as np

def getbaseline():
    data = pd.read_csv('y_train.csv')
    data1=pd.read_csv('X_train.csv')
    a = data[['Sales']]
    m = data1[['Sales']]
    stu1 = np.array(a)
    print(stu1)
    b=len(a)
    sum=0
    print('lenth:',b)
    for i in stu1:
        sum=sum+i
    print('total:',sum)
    average=sum/b
    print('average:',average)
    c=len(m)
    list=[]
    for i in range(c):
         list.append(average)
    array1=np.array(list)
    print(array1)
    return array1
if __name__ == "__main__":
    '''
    This will not be run when imported to another library only if run from this one itself.
    '''
    a=getbaseline()