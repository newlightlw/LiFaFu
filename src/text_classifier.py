import os
import pandas as pd
import numpy as np
import jieba,jieba.analyse
from gensim.models import Word2Vec
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.metrics import precision_score,recall_score,f1_score,roc_curve,auc
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import label_binarize
from sklearn.linear_model.logistic import  LogisticRegression
from sklearn.naive_bayes import  GaussianNB,MultinomialNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

#数据路径
jieba.dt.tmp_dir = os.getcwd()
w2v_filename = "./word2vec_size100_window5_mincount5"
jieba_stopwords = "./stopword.dic"
jieba_userdict = "./user_dict.txt"
ky_path = "./chi2_text_cluster.xls"
train_data_path = "./train_classify.xls"
#数据导入
jieba.load_userdict(jieba_userdict)
jieba.analyse.set_stop_words(jieba_stopwords)
keywords = pd.read_excel(ky_path)
word_name = keywords['word']
word_pro = keywords['max_p']
#算法模型
gs_nb= GaussianNB()
mul_nb = MultinomialNB(alpha=0.5,fit_prior=True)
lg_reg = LogisticRegression(penalty='l2',tol=0.0001,C=1.0,solver='newton-cg',multi_class='multinomial',max_iter=3000)
sgd_clf = SGDClassifier(loss='hinge',penalty='l2',alpha=0.0001,max_iter=1000)
gbt_clf = GradientBoostingClassifier(loss='exponential',learning_rate=0.01,n_estimators=1000,max_depth=7,verbose=1,)
rf_clf = RandomForestClassifier(n_estimators=100,max_features='sqrt',random_state=123,n_jobs=-1,max_depth=7)
weakClassifier=DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5)
adb_clf = AdaBoostClassifier(base_estimator=weakClassifier,algorithm='SAMME',n_estimators=300,learning_rate=0.8)
xgb_clf = XGBClassifier(booster='gbtree',n_estimators=20,learning_rate=0.1,min_child_weight=1,subsample=1,gamma=0.1,objective='multi:softmax',n_jobs=-1,max_depth=10,verbose=1,random_state=123)
lgb_clf = LGBMClassifier(boosting_type='gbdt',objective='multiclass',num_class=9,n_jobs=16,shuffle=True)

class ClassifierModel(object):
    def __init__(self):
        self.w2v = Word2Vec.load(w2v_filename)
        self.word_dict = dict(zip(word_name,word_pro))
        self.clf_models = {'GaussianNB':gs_nb,'MultinomialNB':mul_nb,'LogisticRegression':lg_reg,'SGDClassifier':sgd_clf,
                           'GradientBoostedTrees':gbt_clf,'RandomForest':rf_clf,'AdaBoost':adb_clf,'XGBOOST':xgb_clf,'LGBM':lgb_clf}
    def get_vec(self,sentence):
        vec = np.zeros(self.w2v.vector_size)
        vec_sum = 0
        for word in jieba.cut(str(sentence)):
            try:
                flag = self.word_dict[word]
            except:
                flag = 0
            try:
                if flag > 0.7:
                    vec += self.w2v.wv[word] * flag
                    vec_sum += flag
                else:
                    vec += self.w2v.wv[word] * 0.5
                    vec_sum += 0.5
            except KeyError:
                pass
        return vec if vec_sum == 0 else vec / vec_sum
    def model_metrics(self,model,x,y):
        result = {'f1_score_macro':f1_score(y,model.predict(x),average='macro'),
                  'precision':precision_score(y,model.predict(x),average='macro'),
                  'recall':recall_score(y,model.predict(x),average='macro')
                 }
        return result
    def train(self,datas,labels,model="LGBM",gridsearch=False,parameters=None):
        x_vec = np.array([self.get_vec(i) for i in datas])
        label_uni  = labels.unique()
        num_class = label_uni.size
        label_map = {label:ind for ind,label in enumerate(label_uni)}
        print(label_map)
        _labels = labels.map(label_map)
        print("Original dataset shape %s" % Counter(_labels))
        smote_enn = SMOTEENN(random_state=0)
        x_sample,y_sample = smote_enn.fit_sample(x_vec,_labels)
        print(sorted(Counter(y_sample).items()))
        print('re-sampled dataset shape %s' % Counter(y_sample))
        x_train, x_test, y_train, y_test = train_test_split(x_sample,y_sample,test_size=0.2,random_state=123)
        print("classifier model is:%s" % model)
        y_one_hot = label_binarize(y_test, np.arange(num_class))
        clf_model = self.clf_models[model]
        if gridsearch:
            gsearch = GridSearchCV(clf_model,param_grid=parameters['params'],scoring='accuracy',cv=parameters['cv'],n_jobs=-1)
            gsearch.fit(x_train,y_train)
            print("Best score: %0.3f" % gsearch.best_score_)
            print("Best parameters set:")
            best_parameters = gsearch.best_estimator_.get_params()
            for param_name in sorted(parameters['params'].keys()):
                print("\t%s: %r" % (param_name,best_parameters[param_name]))
            evaluation = self.model_metrics(gsearch.best_estimator_,x_test,y_test)
            print(evaluation)
            y_score = gsearch.predict_proba(x_test)
            fpr,tpr,threshold = roc_curve(y_one_hot.ravel(),y_score.ravel())
            roc_auc = auc(fpr,tpr)
            plt.figure(figsize=(10,10))
            plt.plot(fpr,tpr,color='darkorange',lw=2.0,label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2.0, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('%s ROC curve' % model)
            plt.legend(loc="lower right")
            plt.show()
            joblib.dump(gsearch.best_estimator_,os.path.join(os.getcwd(),model+"gsearch.m"))
        else:
            clf_model.fit(x_train,y_train)
            joblib.dump(clf_model,os.path.join(os.getcwd(),model+".m"))
            evaluation = self.model_metrics(clf_model,x_test,y_test)
            print(evaluation)
            y_score = clf_model.predict_proba(x_test)
            print(y_score - y_one_hot)
            fpr,tpr,threshold = roc_curve(y_one_hot.ravel(),y_score.ravel())
            roc_auc = auc(fpr,tpr)
            plt.figure(figsize=(10,10))
            plt.plot(fpr,tpr,color='darkorange',lw=2.0,label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2.0, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('%s ROC curve' % model)
            plt.legend(loc="lower right")
            plt.show()
            with open("training_evaluation.txt",'w') as f:
                s = model + '\n' + str(evaluation) + str(label_map)
                f.write(s)
model =  ClassifierModel()
data = pd.read_excel(train_data_path)
data = shuffle(data)
datas = data['text']
labels = data['classify']
parameters = {'params':{"n_estimators":[1000,1500,2000,25000,3000],"max_depth":[6,8,10,12,14],"learning_rate":[0.005,0.01,0.05,0.1,0.15]},'cv':3}
model.train(datas,labels,model="LGBM",gridsearch=True,parameters=parameters)

