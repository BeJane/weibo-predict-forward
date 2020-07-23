# -*- coding: utf-8 -*-
'''
Created on 2017年10月31日
@author: Jason.F
'''
import time
import pandas as pd
from sklearn import model_selection
import jieba
import jieba.analyse
 
class cWeibo:
    
    def __init__(self,path):
        self.path=path
 
    def importData(self):
        path=self.path
        #导入样本集
        data=pd.read_csv(path+'train_data.txt',encoding='utf8',sep='\t',names=['luid','mid','time','fcs','ccs','lcs','cont']).astype(str)#nrows=1000
        print(data['fcs'][:200])
        data['fcs']=data['fcs'].astype('int')#博文发表一周后的转发数，权重0.5
        data['ccs']=data['ccs'].astype('int')#博文发表一周后的评论数，权重0.25
        data['lcs']=data['lcs'].astype('int')#博文发表一周后的点赞数，权重0.25
        train,test=model_selection.train_test_split(data,test_size=0.2)
        self.traindata = pd.DataFrame(data)#全量训练
        self.testdata = pd.DataFrame(test)#测试集
        print('训练集，有：', self.traindata.shape[0], '行', self.traindata.shape[1], '列')
        print('测试集，有：', self.testdata.shape[0], '行', self.testdata.shape[1], '列')
        #导入预测集
        data=pd.read_csv(path+'predict_data.txt',encoding='utf8',sep='\t',names=['luid','mid','time','cont']).astype(str)#nrows=100
        self.predata=data #预测集
        print('预测集，有：', self.predata.shape[0], '行', self.predata.shape[1], '列')
    
    def ETL(self):
        
        #uid映射为数字编号
        ut_train=set(self.traindata.iloc[:,0])
        ut_pred=set(self.predata.iloc[:,0])
        ut=list(ut_train.symmetric_difference(ut_pred))#取并集并去重
        df_ut=pd.DataFrame(ut,columns=['luid'])
        df_ut['uid']=df_ut.index
        self.traindata=pd.merge(self.traindata,df_ut, on=['luid'], how='left')
        self.traindata=self.traindata[['uid','mid','time','fcs','ccs','lcs','cont']]
        self.testdata=pd.merge(self.testdata,df_ut, on=['luid'], how='left')
        self.testdata=self.testdata[['uid','mid','time','fcs','ccs','lcs','cont']]
        self.predata=pd.merge(self.predata,df_ut, on=['luid'], how='left')
        self.predata=self.predata[['uid','mid','uid','time','cont']]


        #提取月份（0-11）
        self.traindata['month']=self.traindata.apply(lambda x:(time.strptime(x['time'],"%Y-%m-%d %H:%M:%S")).tm_mon,axis=1)
        
        self.testdata['month']=self.testdata.apply(lambda x:(time.strptime(x['time'],"%Y-%m-%d %H:%M:%S")).tm_mon,axis=1)
        
        self.predata['month']=self.predata.apply(lambda x:(time.strptime(x['time'],"%Y-%m-%d %H:%M:%S")).tm_mon,axis=1)
        # 提取星期（0-6）
        self.traindata['wday']=self.traindata.apply(lambda x:(time.strptime(x['time'],"%Y-%m-%d %H:%M:%S")).tm_wday,axis=1)
        
        self.testdata['wday']=self.testdata.apply(lambda x:(time.strptime(x['time'],"%Y-%m-%d %H:%M:%S")).tm_wday,axis=1)
        
        self.predata['wday']=self.predata.apply(lambda x:(time.strptime(x['time'],"%Y-%m-%d %H:%M:%S")).tm_wday,axis=1)
       
        print(self.traindata)
        #time转换成0-23数字
        self.traindata['time']=self.traindata.apply(lambda x:(time.strptime(x['time'],"%Y-%m-%d %H:%M:%S")).tm_hour,axis=1)
        self.traindata.rename(columns=lambda x:x.replace('time','tid'), inplace=True)#修改列名为tid
        self.testdata['time']=self.testdata.apply(lambda x:(time.strptime(x['time'],"%Y-%m-%d %H:%M:%S")).tm_hour,axis=1)
        self.testdata.rename(columns=lambda x:x.replace('time','tid'), inplace=True)
        self.predata['time']=self.predata.apply(lambda x:(time.strptime(x['time'],"%Y-%m-%d %H:%M:%S")).tm_hour,axis=1)
        self.predata.rename(columns=lambda x:x.replace('time','tid'), inplace=True)    
        print(self.traindata)
        #cont分词，文本内容要考虑带@和红包的特殊意义词
        jieba.suggest_freq('@', True)
        
        self.traindata['cont']=self.traindata.apply(lambda x:",".join(jieba.analyse.extract_tags(x['cont'],topK=50,\
                                        allowPOS=('n','nr','ns','nt','nz','a','ad','an','f','s','i','t','v','vd','vn'))),axis=1)
        self.traindata=self.traindata.drop('cont', axis=1).join(self.traindata['cont'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('tag'))
        print('ok')
        self.testdata['cont']=self.testdata.apply(lambda x:",".join(jieba.analyse.extract_tags(x['cont'],topK=50,\
                                        allowPOS=('n','nr','ns','nt','nz','a','ad','an','f','s','i','t','v','vd','vn'))),axis=1)
        self.testdata=self.testdata.drop('cont', axis=1).join(self.testdata['cont'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('tag'))
        self.predata['cont']=self.predata.apply(lambda x:",".join(jieba.analyse.extract_tags(x['cont'],topK=50,\
                                        allowPOS=('n','nr','ns','nt','nz','a','ad','an','f','s','i','t','v','vd','vn'))),axis=1)
        self.predata=self.predata.drop('cont', axis=1).join(self.predata['cont'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('tag'))
        print('ok')
        #生成标签表
        ft_train=set(self.traindata.iloc[:,8])
        ft_pred=set(self.predata.iloc[:,6])
        #print(self.predata)
        ft=list(ft_train.symmetric_difference(ft_pred))#取并集并去重
        df_ft=pd.DataFrame(ft,columns=['tag'])
        df_ft['fid']=df_ft.index
        df_ft.to_csv("tag.csv")

        print(df_ft[:100])
        self.traindata=pd.merge(self.traindata,df_ft, on=['tag'], how='left')
        self.traindata=self.traindata[['uid','mid','tid','month','wday','fid','fcs','ccs','lcs']]
        self.traindata=self.traindata.dropna(axis=0,how='any')  
        self.traindata['fid']=self.traindata['fid'].astype('int')
        
        self.traindata.to_csv("traindata1.csv")

        self.testdata=pd.merge(self.testdata,df_ft, on=['tag'], how='left')
        self.testdata=self.testdata[['uid','mid','tid','month','wday','fid','fcs','ccs','lcs']]
        self.testdata=self.testdata.dropna(axis=0,how='any') 
        self.testdata['fid']=self.testdata['fid'].astype('int')   

        self.testdata.to_csv("testdata1.csv")       
        self.predata=pd.merge(self.predata,df_ft, on=['tag'], how='left')
        self.predata=self.predata[['uid','mid','tid','month','wday','fid']]
        self.predata=self.predata.dropna(axis=0,how='any')
        self.predata['fid']=self.predata['fid'].astype('int')
        self.predata.to_csv("predata1.csv")

wb=cWeibo('F:\\大三下\\数据挖掘\\微博\\')
wb.importData()#导入数据
#wb.ETL()#特征抽取