#encoding=utf-8
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR,LinearSVC,SVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,GridSearchCV 
from sklearn import metrics
from sklearn.metrics import accuracy_score
import random
import math
import copy

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

x = pd.read_csv('jbxx_out.csv',encoding='gbk')
def onehot(X):
    enc = OneHotEncoder()
    X=enc.fit_transform(X.values).toarray()
    return np.transpose(X)
jbxx_info_1 = pd.DataFrame(columns=['cj','kjlw','aszl']  )
######used for regression########
array_temp = pd.concat([x['YXSM'],x['KSFS'],x['PYFSM'],x['XSLBM'] ], axis =1)#yuanxi kaoshi peiyangfnagshi xueshengleibie
index = np.arange(len(x))
tem_frame=pd.DataFrame(np.transpose(onehot(array_temp)) ,index=index)
#############################
###########scale grade####################
jbxx_info_1['cj'] = x['cj']
jbxx_info_1['kjlw'] = x['KJLW']
jbxx_info_1['aszl'] = x['ASZL']#0.653167287324
 
X = pd.concat([tem_frame,jbxx_info_1],axis=1)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X=pd.DataFrame(X)
y = x['bi_class']#0.65762425895288679
X_dev, X_test, Y_dev, Y_test = train_test_split(X, y, random_state=1)
print(Y_dev)
print(Y_test)
def stacking():
    n_trees = 10
    n_folds = 3 
    # Our level 0 classifiers
    clfs = [
        #RandomForestClassifier(n_estimators = n_trees, criterion = 'gini'),
        #AdaBoostClassifier(n_estimators = n_trees * 2 ),
        #GradientBoostingClassifier(n_estimators = n_trees),
        XGBClassifier(max_depth=100, learning_rate=0.02, n_estimators=200, \
                                  silent=True, objective='binary:logistic', nthread=-1, \
                                  gamma=0, min_child_weight=1, max_delta_step=0, \
                                  subsample=1, colsample_bytree=1, colsample_bylevel=1, \
                                  reg_alpha=1, reg_lambda=1, scale_pos_weight=1, \
                                  base_score=0.6, seed=random_seed, missing=None)
    ]
    
    # Ready for cross validation
    skf = list(StratifiedKFold(Y_dev, n_folds))
    
    # Pre-allocate the data
    blend_train = np.zeros((X_dev.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs))) # Number of testing data x Number of classifiers

    for j, clf in enumerate(clfs):
        print ('Training classifier [%s]' % (j))
        blend_test_j = np.zeros((X_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            print ('Fold [%s]' % (i))
            
            # This is the training and validation set
            X_train = X_dev.iloc[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev.iloc[cv_index]
            Y_cv = Y_dev[cv_index]
            Y_train.fillna(0, inplace=True)
            clf.fit(X_train, Y_train)
            
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[cv_index, j] = clf.predict(X_cv)
            blend_test_j[:, i] = clf.predict(X_test)
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)
    
    print ('Y_dev.shape = %s' % (Y_dev.shape))  
    # Start blending!
    bclf = LogisticRegression()
    bclf.fit(blend_train, Y_dev)
    
    # Predict now
    Y_test_predict = bclf.predict(blend_test)
    print( Y_test_predict)
    score = metrics.accuracy_score(Y_test, Y_test_predict)
    print ('Accuracy = %s' % (score))  
    return score
###############regression#########
def lr():
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    y_pred = pd.Series(y_pred)
    print ("MAE:",metrics.mean_absolute_error(y_test, y_pred))
##############lr: 0.72412410637056845#########################
def svr():
    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict( X_test)
    print("MAE:",metrics.mean_absolute_error(y_test, y_pred))
############ svr:0.73431758213792031######
def ada():
    clf = AdaBoostClassifier(n_estimators=100)
    regr_1 = DecisionTreeRegressor(max_depth=4)
    rng = np.random.RandomState(1)
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)
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
    plt.scatter(X, y_test, c="k", label="training samples")
    plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
    plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show()
#ada()
############################classification################### 
def target_2_class():
    T_F = jbxx_info['bynx']-jbxx_info['XZ']
    graduate_dict = {}
    graduated = T_F[T_F>=0]
    for x in graduated:
        graduate_dict[x] = 0
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
    ###########cross validation############
    n_estimators = [50, 100, 150, 200]
    max_depth = [2, 4, 6, 8]
    colsample_bytree=[0.2,0.3,0.5,0.8,1.0]
    learning_rate=[0.01,0.02,0.03,0.04,0.05]
    #Best: -0.347072 using {'learning_rate': 0.05, 'max_depth': 4, 'colsample_bytree': 1.0, 'n_estimators': 200}
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators,colsample_bytree=colsample_bytree,learning_rate=learning_rate)
    model = XGBClassifier(max_depth=4, learning_rate=learning_rate, n_estimators=200, \
                              silent=True, objective='binary:logistic', nthread=-1, \
                              gamma=0, min_child_weight=1, max_delta_step=0, \
                              subsample=1, colsample_bytree=colsample_bytree, colsample_bylevel=1, \
                              reg_alpha=1, reg_lambda=1, scale_pos_weight=1, \
                              base_score=0.5, seed=random_seed, missing=None)
    model.fit(X_train, y_train)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
    result = grid_search.fit(X_train, y_train)
    y_pred=model.predict(X_pre)
    y_pred = pd.Series(y_pred)
    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    y_pred = pd.DataFrame(y_pred)
    y_pred['XH'] = X_pre['XH']
    y_pred.to_csv("prediction.csv")
    print(metrics.classification_report(y_train, y_pred))
    edictions = [round(value) for value in y_pred]
    #####evaluate predictions#########
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
def rf():
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    y_pred = pd.Series(y_pred)
    print(metrics.classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

def get_xgb_feat_importances(clf):
    if isinstance(clf, xgb.XGBModel):
        # clf has been created by calling
        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
        fscore = clf.booster().get_fscore()
    else:
        # clf has been created by calling xgb.train.
        # Thus, clf is an instance of xgb.Booster.
        fscore = clf.get_fscore()
    feat_importances = []
    for ft, score in fscore.iteritems():
        feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(
        by='Importance', ascending=False).reset_index(drop=True)
    # Divide the importances by the sum of all importances
    # to get relative importances. By using relative importances
    # the sum of all importances will equal to 1, i.e.,
    # np.sum(feat_importances['importance']) == 1
    feat_importances['Importance'] /= feat_importances['Importance'].sum()
    # Print the most important features and their importances
    print (feat_importances.head())
    return feat_importances

if __name__ == '__main__':
    best_score = 0.0   
    # run many times to get a better result, it's not quite stable.
    for i in range(1):
        print ('Iteration [%s]' % (i))
        score = stacking()
        best_score = max(best_score, score)        
    print ('Best score = %s' % (best_score)) 