# SVM-stcock-pick
mechine learning


```python
import pandas as pd
import numpy as np

path_factor ='/Users/44091781/Desktop/Python Test/Machine Learning/2015-2017-factor.xlsx'df = pd.read_excel(path_factor,sheetname = "2015-2017-factor data")
num = 50
list = df.columnslist_stock = []for i in range(num):    list_stock.append(list[i*16])
list_ratio = pd.read_excel(path_factor,sheetname = "Sheet1")

#(1) 筛选 factor，剔出相关性强的 
data = df.iloc[:,:]k = np.linspace(1,14,14)for i in range(num):    data.columns.values[i*16:i*16+14] = k    
cov_factor = data.iloc[:,0:14]
cov_factor = cov_factor.astype(float)
check_cor = cov_factor.corr()

list_drop=[]
for i in range(len(check_cor)): 
   if check_cor.iloc[0,i]>0.6:
      list_drop.append(check_cor.columns[i])
      
def label_c_s(data_factor,i):
   data_factor['stockcode'] = list_stock[i]
   data_factor['return']= np.log(data_factor.iloc[:,0]/data_factor.iloc[:,0].shift(1))
   data_factor= data_factor.iloc[1:,:]
   data_factor['status'] = 1
   judge = data_factor.iloc[:,0].fillna(999)
   for i in range(len(judge)):
      if judge.iloc[i] == 999:
         data_factor['status'][i] = 0
      data_factor = data_factor.drop(list_drop,axis=1)
      return data_factor
      
   #用price judege NAN，不是因子，观测是否会有影响
data_factor = data.iloc[:,:14]
data_factor = label_c_s(data_factor,0)
for i in range(49):
   i=i+1
   data_factor2 = data.iloc[:,i*16:i*16+14]
   data_factor2 = label_c_s(data_factor2,i)
   data_factor = pd.concat([data_factor,data_factor2],axis=0)
train_data = data_factor.sort_index(axis=0,ascending=True)
