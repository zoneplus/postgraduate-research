#encoding=utf-8
import pandas as pd

import copy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score
import random
import math

random_seed = 0
random.seed(random_seed)

x = pd.read_csv('jbxx_out.csv',encoding='gbk')

def onehot(X):
    enc = OneHotEncoder()
    X = enc.fit_transform(X.values).toarray()
    return np.transpose(X)

jbxx_info_1 = pd.DataFrame(columns=['cj','kjlw','aszl']  )
 ######used for regression########
array_temp = pd.concat([x['YXSM'],x['KSFS'],x['PYFSM'],x['XSLBM'] ], axis =1)#yuanxi kaoshi peiyangfnagshi xueshengleibie
index = np.arange(len(x))
tem_frame=pd.DataFrame(np.transpose(onehot(array_temp)) ,index=index)
###########scale grade####################

jbxx_info_1['cj'] = x['cj']
jbxx_info_1['kjlw'] = x['KJLW']
jbxx_info_1['aszl'] = x['ASZL']#0.653167287324
  
X = pd.concat([tem_frame,jbxx_info_1],axis=1)
scaler = StandardScaler().fit(X)

X = scaler.transform(X)
X = pd.DataFrame(X)
y = x['bi_class']#0.65762425895288679
print(X.columns)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
cv = StratifiedShuffleSplit(n_splits=4, test_size=0.1, random_state=42)
for train_index, test_index in cv.split(X, y):
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]#for dataframe get sample with iloc;for nparray,with X[]
	y_train, y_test = y[train_index], y[test_index]

	X_train.fillna((-999), inplace=True) 
	X_test.fillna((-999), inplace=True)

	#######################################regression##########################################################
	def lr():
	    linreg = LinearRegression()
	    linreg.fit(X_train, y_train)
	    y_pred = linreg.predict(X_test)
	    y_pred = pd.Series(y_pred)
	    print(y_pred)
	    print(y_test)
	    print ("MAE:",metrics.mean_absolute_error(y_test, y_pred))

	def svr():
	    clf = SVR(C=1.0, epsilon=0.2)
	    clf.fit(X_train, y_train)
	    y_pred = clf.predict( X_test)
	    print("MAE:",metrics.mean_absolute_error(y_test, y_pred))

	def ada():
	    clf = AdaBoostClassifier(n_estimators=100)
	    regr_1 = DecisionTreeRegressor(max_depth=4)
	    rng = np.random.RandomState(1)
	    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)
	    regr_1.fit(X_train, y_train)
	    regr_2.fit(X_train, y_train)
	    # Predict
	    y_1 = regr_1.predict(X_test)
	    y_2 = regr_2.predict(X_test)

	    print("MAE_dc:",metrics.mean_absolute_error(y_test,  y_1))# 0.73552208072997316
	    print("MAE_ada:",metrics.mean_absolute_error(y_test,  y_2))#0.75
	    # Plot the results
	    plt.figure()
	    print(X_test.shape)#3892*50
	    X = np.arange(755)
	    plt.scatter(X, y_test, c = "k", label = "training samples")
	    plt.plot(X, y_1, c = "g", label = "n_estimators=1", linewidth=2)
	    plt.plot(X, y_2, c = "r", label = "n_estimators=300", linewidth=2)
	    plt.xlabel("data")
	    plt.ylabel("target")
	    plt.title("Boosted Decision Tree Regression")
	    plt.legend()
	    plt.show()
	#ada()
	#######################################classification##############################################################

	def target_2_class():
	    T_F = jbxx_info['bynx']-jbxx_info['XZ']
	    graduate_dict = {}
	    graduated = T_F[T_F>=0]
	    for x in graduated:
	        graduate_dict[x] = 0
	    #gradauted.apply(lambda x:)
	    print(graduate_dict)
	    ungraduated = T_f[T_F<0]
	    for x in ungraduated:
	        graduate_dict[x] = 1
	    return T_F
	#CLASS:{-1，0，1，2，3，4，5}
	def dt():
	    clf = DecisionTreeClassifier(random_state=0)
	    clf = clf.fit(X_train, y_train)
	    y_pred = clf.predict( X_test)
	    y_pred = pd.Series(y_pred)
        print(metrics.classification_report(y_test, y_pred))
	    print("MAE:",metrics.mean_absolute_error(y_test, y_pred))
	    
	def xgb():
	    #n_estimators = [50, 100, 150, 200]
	    #max_depth = [2, 4, 6, 8]
	    #colsample_bytree=[0.2,0.3,0.5,0.8,1.0]
	    #learning_rate=[0.01,0.02,0.03,0.04,0.05]
	    #Best: -0.347072 using {'learning_rate': 0.05, 'max_depth': 4, 'colsample_bytree': 1.0, 'n_estimators': 200}
	    #param_grid = dict(max_depth=max_depth, n_estimators=n_estimators,colsample_bytree=colsample_bytree,learning_rate=learning_rate)
	    model = XGBClassifier(max_depth=70, learning_rate=0.09, n_estimators=300, \
	                              silent=True, objective='binary:logistic', nthread=-1, \
	                              gamma=0, min_child_weight=1, max_delta_step=0, \
	                              subsample=1, colsample_bytree=1, colsample_bylevel=1, \
	                              reg_alpha=1, reg_lambda=1, scale_pos_weight=1, \
	                              base_score=0.5, seed=random_seed, missing=None)
	    model.fit(X_train, y_train)
	    y_pred = model.predict(X_test)
	    y_pred = pd.Series(y_pred)
	    y_pred = pd.DataFrame(y_pred)
        score =  f1_score(y_test, y_pred)
	    accscore =  accuracy_score(y_test, y_pred)
	    print ('accuracy score:     %0.3f' % accscore)
	    print("f1-score:   %0.3f" % score)      
	    print(metrics.classification_report(y_test, y_pred))
	    
	def rf():
	    model = RandomForestClassifier(n_estimators=20)
	    model.fit(X_train, y_train)
	    y_pred=model.predict(X_test)
	    y_pred = pd.Series(y_pred)
	    print(metrics.classification_report(y_test, y_pred))
	    accuracy = accuracy_score(y_test, y_pred)
	    print("Accuracy: %.2f%%" % (accuracy * 100.0))

	def get_xgb_feat_importances(clf):
	    if isinstance(clf, xgb.XGBModel):
	        fscore = clf.booster().get_fscore()
	    else:
	        fscore = clf.get_fscore()
	    feat_importances = []
	    for ft, score in fscore.iteritems():
	        feat_importances.append({'Feature': ft, 'Importance': score})
	    feat_importances = pd.DataFrame(feat_importances)
	    feat_importances = feat_importances.sort_values(
	        by='Importance', ascending=False).reset_index(drop=True)
	    feat_importances['Importance'] /= feat_importances['Importance'].sum()
	    # Print the most important features and their importances
	    print (feat_importances.head())
	    return feat_importances
	if __name__=='__main__':
	    rf() 