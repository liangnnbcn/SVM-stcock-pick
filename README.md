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

#%%
length = len(df)
class Para:
    method = 'svm'
    month_in_sample = range(82,153+1)
    month_in_test = range(154, 230+1)
    percent_select = [0.3, 0.3]
    percent_cv = 0.1
    path_data = '/Users/44091781/Desktop/Python Test/Machine Learning'
    path_results = '/Users/44091781/Desktop/Python Test/Machine Learning'
    seed = 42
    svm_kernel = 'linear'
    svm_c = 0.01 #SVM惩罚系数C
    para = Para()
def label_data(data):
    data['return_bin'] = np.nan
    data = data.sort_values (by= 'return' , ascending = False)
    n_stock_positive= int(para.percent_select[0]*len(data))
    n_stock_negative= -int(para.percent_select[1]*len(data))
    data.iloc[0: n_stock_positive,-1] = 1
    data.iloc[n_stock_negative:,-1] = 0
    data = data.dropna(axis=0)
    return data
 
 for i_month in para.month_in_sample:    data_curr_month = train_data.iloc[i_month*50:i_month*50+50,:].dropna(axis = 0)    data_curr_month = label_data(data_curr_month)    if i_month == para.month_in_sample[0]:        data_in_sample = data_curr_month    else:        data_in_sample = data_in_sample.append(data_curr_month)
 
 X_in_sample = data_in_sample.iloc[:,0:9]     y_in_sample = data_in_sample.iloc[:,-1]from sklearn.model_selection import train_test_splitX_train,X_cv,y_train,y_cv=train_test_split(X_in_sample,y_in_sample,test_size=para.percent_cv,random_state=para.seed)
 
 from sklearn import decomposition #主成分分析pca = decomposition.PCA(n_components = 0.95)pca.fit(X_train)X_train = pca.transform(X_train)X_cv = pca.transform(X_cv)
 
 ```
 
 

#%% 训练


'''regression'''


y_in_sample = data_in_sample.loc[:,'return']


from sklearn import preprocessing


scaler = preprocessing.StandardScaler().fit(X_train)


X_train = scaler.transform(X_train)


X_cv = scaler.transform(X_cv) #将原数据转换为均值为0，标准差为1 的新数据。


'''Linear regression'''


if para.method =='LR':


    from sklearn
import linear_model


    model =
linear_model.LinearRegression(fit_intercept=True)


'''Linear regression训练'''


if para.method == 'LR':


   
model.fit(X_train,y_train)


    y_score_train =
model.decision_function(X_train)


    y_score_cv =
model.decision_function(X_cv)


  


'''SVM'''


if para.method =='SVM':


    from sklearn
import svm


    model =
svm.SVC(kernel = para.svm_kernel,C=para.svm_c)


'''SVM 训练'''


if para.method == 'SVM':


   
model.fit(X_train,y_train)


    y_pred_train =
model.predict(X_train)


    y_score_train =
model.decision_function(X_train)


    y_pred_cv =
model.predict(X_cv)


    y_score_cv =
model.decision_function(X_cv)


 


'''SGD + hinge'''


 


from sklearn import svm


model = svm.SVC(kernel = para.svm_kernel,C=para.svm_c)    


model.fit(X_train,y_train)   



y_pred_train = model.predict(X_train)


y_score_train = model.decision_function(X_train)  


y_pred_cv = model.predict(X_cv)  


y_score_cv = model.decision_function(X_cv)


 


#回头可以补充


#%% 预测  ，


'''创建三个空数据集y_true_test、y_pred_test 和y_score_test'''


y_true_test = pd.DataFrame([np.nan] * np.ones((num,1)))


y_true_test.index = list_stock


y_pred_test = pd.DataFrame([np.nan] * np.ones((num,1)))


y_pred_test.index = list_stock


y_score_test = pd.DataFrame([np.nan] * np.ones((num,1)))


y_score_test.index = list_stock


 


#因子NAN也算停牌，status只是价格停牌


for i_month in para.month_in_test:


    data_curr_month =
train_data.iloc[i_month*50:i_month*50+50,:]        


   
data_curr_month_dropna = data_curr_month.dropna(axis = 0)


    X_curr_month =
data_curr_month_dropna.iloc[:,0:9]


    X_curr_month =
pca.transform(X_curr_month)


    


    y_pred_curr_month
= model.predict(X_curr_month)


    y_score_curr_month
= model.decision_function(X_curr_month)


    


    y_true_df =
data_curr_month['return']


    y_true_df.index =
data_curr_month['stockcode']


    y_pred_df =
pd.DataFrame(y_pred_curr_month)


    y_pred_df.index =
data_curr_month_dropna['stockcode']


    y_score_df =
pd.DataFrame(y_score_curr_month)


    y_score_df.index =
data_curr_month_dropna['stockcode']


    


    y_true_test =
pd.concat([y_true_test,y_true_df],axis=1)


    y_pred_test =
pd.concat([y_pred_test,y_pred_df],axis=1)


    y_score_test =
pd.concat([y_score_test,y_score_df],axis=1)


 


y_true_test = y_true_test.iloc[:,1:]


y_pred_test = y_pred_test.iloc[:,1:]


y_score_test = y_score_test.iloc[:,1:]


 


#%% 评价


from sklearn import metrics


print('training set, accuracy =
%.2f'%metrics.accuracy_score(y_train, y_pred_train))


print('training set, AUC = %.2f'%metrics.roc_auc_score(y_train,
y_score_train))


print('cv set, accuracy = %.2f'%metrics.accuracy_score(y_cv,
y_pred_cv))


print('cv set, AUC = %.2f'%metrics.roc_auc_score(y_cv,
y_score_cv))


 


y_true_total = pd.DataFrame([np.nan] * np.ones((1,1)))


y_pred_total = pd.DataFrame([np.nan] * np.ones((1,1)))


y_score_total = pd.DataFrame([np.nan] * np.ones((1,1)))


 


a= 0


for i_month in para.month_in_test:


    


    data_curr_month =
train_data.iloc[i_month*50:i_month*50+50,:]        


   
data_curr_month_dropna = data_curr_month.dropna(axis = 0)


   
data_curr_month_dropna.index = data_curr_month_dropna['stockcode']


    y_curr_month =
label_data(data_curr_month_dropna)['return_bin']


    


    y_pred_curr_month
= y_pred_test.iloc[:,a]


    y_score_curr_month
= y_score_test.iloc[:,a]


 


    y_pred_curr_month
= y_pred_curr_month[y_curr_month.index]


    y_score_curr_month
= y_score_curr_month[y_curr_month.index]


    


    y_true_total =
pd.concat([y_true_total,y_curr_month],axis=0)


    y_pred_total =
pd.concat([y_pred_total,y_pred_curr_month],axis=0)


    y_score_total =
pd.concat([y_score_total,y_score_curr_month],axis=0)


    a = a+1


    


    print('test set,
month %d, accuracy = %.2f'%(i_month,metrics.accuracy_score(y_curr_month,
y_pred_curr_month)))    


    print('test set,
month %d, AUC = %.2f'%(i_month,
metrics.roc_auc_score(y_curr_month,y_score_curr_month)))


 


y_true_total = y_true_total.iloc[1:,:]


y_pred_total = y_pred_total.iloc[1:,:]


y_score_total = y_score_total.iloc[1:,:]


print('total test set, accuracy =
%.2f'%metrics.accuracy_score(y_true_total, y_pred_total))


print('total test set, AUC =
%.2f'%metrics.roc_auc_score(y_true_total, y_score_total))


    


#%% 构建策略  ==等权


para.n_stock_select = 3


length = para.month_in_test[-1] - para.month_in_test[0]


strategy = pd.DataFrame({'return':[0] * length,'value':[1] *
length,


                        
'benchmark-r':[0]* length,'benchmark-v':[1]* length})   


a = 0


for i_month in para.month_in_test:    


    y_true_curr_month
= y_true_test.iloc[:,a]


    y_score_curr_month
= y_score_test.iloc[:,a]


    y_score_curr_month
= y_score_curr_month.sort_values(ascending=False)


    index_select =
y_score_curr_month[0:para.n_stock_select].index


   
strategy.loc[a,'return'] = np.mean(y_true_curr_month[index_select])


    strategy['value']
= (strategy['return']+1).cumprod()


   
strategy.loc[a,'benchmark-r'] = np.mean(y_true_curr_month)


   
strategy['benchmark-v'] = 
(strategy['benchmark-r']+1).cumprod()


    a=a+1


    


import matplotlib.pyplot as plt


plt.plot(strategy.index,strategy['value'],'r-')


plt.show()


 


plt.figure(2,figsize=(10, 6)) 


lines1 =
plt.plot(strategy.index,strategy['value'],label='svm')


lines2 =
plt.plot(strategy.index,strategy['benchmark-v'],label='equal index ')


plt.grid(True)  


plt.legend( loc = "upright")


 


ann_excess_return = np.mean(strategy['return']) * 251


ann_excess_vol = np.std(strategy['return']) * np.sqrt(251)


info_ratio = ann_excess_return/ann_excess_vol


 


ann_excess_return2 = np.mean(strategy['benchmark-r']) * 251


ann_excess_vol2 = np.std(strategy['benchmark-r']) *
np.sqrt(251)


info_ratio2 = ann_excess_return/ann_excess_vol


 


print('annual excess return = %.2f'%ann_excess_return)


print('annual excess volatility = %.2f'%ann_excess_vol)


print('information ratio = %.2f'%info_ratio)


