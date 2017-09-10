#encoding=utf-8
import pandas as pd
from datetime import datetime
import time
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns

#jbxx_info2 = pd.read_excel('data2xlsx',encoding='GBK') 
kaiti = pd.read_csv('C:\\Users\\qiqi\\Desktop\\data2.csv',encoding='GBK') 
#byn = pd.read_csv('C:\\Users\\qiqi\\Desktop\\jbxx_out2.csv',encoding='GBK') 
#k = pd.merge(kaiti,byn,on="XH")
#k.to_csv("merged.csv",encoding="gbk")
print(kaiti.corr())
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
def kaitishijian():
    plt.figure(figsize=(8, 8))
    y =  kaiti.groupby('gap').bi_class.mean()
    plt.scatter(jbxx_info['bynx'], y,color='purple')
    plt.ylabel(u"延期时间")
    plt.xlabel(u"开题时间")
    plt.title(u"开题时间与延期时间的关系")
    kaiti.gap.value_counts().plot(kind='bar',color='purple')
    plt.rcParams['font.sans-serif']=['SimHei']  
    plt.show()
def statichunyin():
    sizes = jbxx_info.groupby('XBM').bynx.mean()
    print (sizes)
    labels = [u'男生',u'女生']
    colors = ['lightskyblue','lightcoral']
    explode = (0,0)
    plt.bar([1,2],sizes,width=0.8,facecolor = 'lightskyblue',edgecolor = 'white')  
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.tight_layout()
    plt.show()
def staticstation():
    xb = jbxx_info.groupby('SYD').size()
    plt.figure(figsize=(6,6))
    labels = [u'未婚',u'已婚',u'离婚']
    sizes = xb
    colors = ['gold','lightcoral','lightskyblue']
    explode = (0,0,0)
    patches,l_text,p_text = plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow = False, startangle=140)
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.legend(patches, labels, loc="best")
    plt.tight_layout()
    plt.show()
def zhengzhimianmao():
    xb = jbxx_info.groupby('ZZMM').bynx.mean()
    plt.figure(figsize=(6,6))
    plt.title(u'不同政治面貌的博士生平均毕业年限')
    plt.xticks((1,2,3,4,5,6,7),(u'中共党员',u'中共预备党员',u'共青团员',u'民革会员',u'民盟盟员',u'九三学社社员',u'群众'))
    dic={'1':1,'2':2,'3':3,'4':4,'5':5,'10':6,'13':7}
    xb_num = []
    for i in xb.index:
        xb_num.append(dic[str(i)])   
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.tight_layout()
    plt.xlabel(u'政治面貌')  
    plt.ylabel(u'平均延期时间')  
    plt.bar(xb_num,xb,width=0.35,facecolor = 'lightskyblue',edgecolor = 'white')  
    plt.show()
def scatter_yuanxi():   
    plt.figure(figsize=(6,6))
    plt.title(u'不同院系的博士延期时间统计')  
    #设置X轴标签  
    plt.xlabel(u'院系')  
    #设置Y轴标签  
    plt.ylabel(u'延期时间')  
    x = ['1','2','3','4','5','6','7','8','9','11','13','14','15','17','27']
    x = map(int,x)
    y = jbxx_info.groupby('YXSM').bynx.mean()
    plt.xlim(0,28)
    plt.ylim(0,3)
    plt.xticks(x)
    plt.scatter(x,y,c = 'g',marker = 'o') 
    plt.rcParams['font.sans-serif']=['SimHei']  
    plt.show()   
 
def cj(): 
    plt.figure(figsize=(8, 8))
    y = jbxx_info['cj'] 
    print (jbxx_info.groupby('bynx'))
    plt.scatter(jbxx_info['bynx'], y,color='purple')
    plt.ylabel(u"平均成绩")
    plt.xlabel(u"毕业时间")
    plt.title(u"学生成绩与毕业时间的关系")
    plt.rcParams['font.sans-serif']=['SimHei']  
    plt.show()
def scatter_leibie():
    plt.figure(figsize=(6,6))
    plt.title(u'不同类别的博士平均延期时间')  
    #设置X轴标签  
    plt.xlabel(u'学生类别')  
    #设置Y轴标签  
    plt.ylabel(u'平均延期时间')  
    y = jbxx_info.groupby('XSLBM').bynx.mean()
    plt.xticks((1,2,3,4,5),(u'普通博士',u'直博生',u'提前攻博',u'硕博连读',u'高研班'))
    dic={'110.0':1,'111.0':2,'112.0':3,'113.0':4,'114.0':5}
    xb_num = []
    for i in y.index:
        xb_num.append(dic[str(i)]) 
    rect = plt.bar(xb_num,height = y,facecolor = 'lightskyblue',width=0.55,edgecolor = 'white')
    plt.rcParams['font.sans-serif']=['SimHei']  
    plt.show() 

def feature_extract():
    jbxx_info['byzy_dalei'] = pd.Series()
    jbxx_info['age'] = pd.Series()
    for i in jbxx_info.index:
        jbxx_info['age'].loc[i] = datetime((jbxx_info['RXNY'].astype(str).apply(lambda x:x[:4])).astype(int).loc[i],(jbxx_info['RXNY'].astype(str).apply(lambda x:x[4:6])).astype(int).loc[i],1)-datetime((jbxx_info['CSRQ'].astype(str).apply(lambda x:x[:4])).astype(int).loc[i],(jbxx_info['CSRQ'].astype(str).apply(lambda x:x[4:6])).astype(int).loc[i],1)
        jbxx_info['age'].loc[i] = pd.to_timedelta(jbxx_info['age'][i],unit='d').total_seconds()/(60*60*24*365)
    jbxx_info.to_csv('graduate/out4.csv',index = False, float_format = '%.2f',encoding='gbk')

def JG():
    xb = jbxx_info.groupby('sfjh').bynx.mean()
    plt.figure(figsize=(6,6))
    plt.title(u'婚姻状况与博士生平均毕业年限的关系')
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.tight_layout()
    plt.xlabel(u'是否结婚')
    plt.ylabel(u'平均毕业延期年限')
    plt.xticks((1,2,3),(u'未婚',u'已婚',u'离婚'))
    rect = plt.bar(xb.index,height = xb,facecolor = 'lightskyblue',width=0.2,edgecolor = 'white')

    plt.show()
def KSFS():
    xb = jbxx_info.groupby('KSFS').bynx.mean()
    plt.figure(figsize=(6,6))
    plt.title(u'入学考试方式与博士生平均毕业延期年限的关系')
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.tight_layout()
    plt.xlabel(u'考试方式')
    plt.ylabel(u'平均毕业延期年限')
    plt.xticks((11,12,13,14,15,16,17,18),(u'公招',u'推免',u'硕博连读',u'本科毕业生直博',u'硕士生提前攻博',u'硕士生推荐免试',u'硕士生单独考试',u'高研中心'))
    dic={'11':11,'12':12,"13":13,"17":14,"18":15,"22":16,"23":17,"40":18}
     
    xb_num = []
    for i in xb.index:
        xb_num.append(dic[str(i)]) 
    rect = plt.bar(xb_num,height = xb,facecolor = 'lightskyblue',width=0.55,edgecolor = 'white')
    plt.show()
def PYFS():
    xb = jbxx_info.groupby('PYFSM').bynx.mean()
    #调节图形大小，宽，高
    fig = plt.figure(figsize=(20,10))
    plt.title(u'培养方式与博士生平均毕业延期年限的关系')
    plt.rcParams['font.sans-serif']=['SimHei'] 
    #plt.legend(patches, labels, loc="best")
    plt.tight_layout()
    plt.xlabel(u'培养方式')
    plt.ylabel(u'平均毕业延期年限')
    plt.xticks((11,12,13,14,15),(u'非定向',u'定向',u'委托培养',u'自筹经费',u'其他'))
    dic={'11':11,'12':12,'23':13,'24':14,'29':15}
    xb_num = []
    for i in xb.index:
        xb_num.append(dic[str(i)]) 
    rect = plt.bar(xb_num,height = xb,facecolor = 'lightskyblue',width=0.55,edgecolor = 'white')
    plt.show()
def mapf(a,b):
    return (a,b)
def Daoshi():
    xb = jbxx_info.groupby('daoshi').bynx.mean()
    fig = plt.figure(figsize=(20,10))
    plt.title(u'导师与博士生平均毕业延期年限的关系')
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.xlabel(u'导师工作证号')
    plt.ylabel(u'平均毕业延期年限')
    l = len(set(jbxx_info['daoshi']))
    ###旋转刻度值
    axis = plt.gca().xaxis 
    plt.setp( axis.get_majorticklabels(), rotation=90 ) 
    t = list(set(jbxx_info['daoshi'].get_values())) 
    t_trans = map(str,t)
    plt.xticks((range(l) ),(map(lambda x:x.zfill(5),t_trans)))  
    d = []
    for j in t:
        d.append(j)
    dic = dict(map(mapf,d,range(l)  ))
    xb_num = []
    for i in xb.index:
        xb_num.append(dic[i]) 
    rect = plt.scatter(xb_num,xb,c = 'g',marker = 'o')
    left = 0.05
    bottom = 0.45 
    width = 0.2
    height = 0.2
    plt.show()
def cj():
    xb = jbxx_info.groupby('KJLW').bynx.mean()
    plt.figure(figsize=(8, 8))
    plt.scatter(range(10),xb,color='purple')
    plt.ylabel(u"科技论文数目")
    plt.xlabel(u"毕业时间")
    plt.title(u"科技论文数目与毕业时间的关系")
    plt.rcParams['font.sans-serif']=['SimHei']  
    plt.show()
def overview():
    xb = jbxx_info['bynx_int'].value_counts()
    plt.figure(figsize=(6,6))
    plt.title(u'博士生平均毕业延期年限的分布')
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.tight_layout()
    plt.xlabel(u'平均毕业延期年限')
    plt.ylabel(u'人数')
    rect = plt.bar(range(5),height = xb,facecolor = 'lightskyblue',width=0.55,edgecolor = 'white')
    plt.show()
def syd():
    xb = jbxx_info.groupby('SYD').bynx.mean()
    plt.figure(figsize=(6,6))
    plt.title(u'不同生源地博士生平均毕业延期年限')
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.tight_layout()
    plt.ylabel(u'平均毕业延期年限')
    plt.xlabel(u'生源地')
    jbxx_info2['pro'] = jbxx_info2['pro'].map(lambda x:x.encode('utf-8'))
    mapstr = map(str,jbxx_info2['pro'])
    plt.xticks((range(30)[1:]),(mapstr))
    dic={'11':1,'12':2,'13':3,'14':4,'15':5,'21':6,'22':7,'23':8,'31':9,'32':10,'33':11,'34':12,'35':13,'36':14,'37':15,'41':16,'42':17,'43':18,'44':19,'45':20,'46':21,'50':22,'51':23,'52':24,'53':25,'61':26,'62':27,'64':28,'65':29}
    xb_num = []
    for i in xb.index:
        xb_num.append(dic[str(i)]) 
    rect = plt.bar(xb_num,height = xb,facecolor = 'lightskyblue',width=0.55,edgecolor = 'white')
    plt.show()
 
