import lightgbm as lgb
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText
#import logging
import jieba, jieba.analyse
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import warnings
from Smote import Smote
from imblearn.combine import SMOTEENN

#综合采样：{'LightGBM': {'f1-score-macro': 0.9087153319023766, 'precision': 0.9072374663030269, 'recall': 0.9148032168505825, 'f1-score-micro': 0.91}}
#Smote: {'LightGBM': {'f1-score-macro': 0.867329178128796, 'precision': 0.8689650204001673, 'recall': 0.8686064017428986, 'f1-score-micro': 0.866194411648957}}

def __init__(self):
    self.threshold = 2000
    
def load_w2v(f_name):
#    model = Word2Vec.load(f_name)
    model = FastText.load(f_name)
#    stopwords = r'D:\kst\work\libs\stopword_nanke.dic'
    stopwords = r'D:\kst\work\libs\stopword.dic'
    userdict = r'D:\kst\work\libs\binkes_dic_2018_06_15t.txt'
    # userdict = '../data/user_dict_n.txt'

    jieba.analyse.set_stop_words(stopwords)  # 导入停用词
    jieba.load_userdict(userdict)  # 导入自定义词典
    return model

##分词
def get_vec(sentence, model):
    vec = np.zeros(model.vector_size)
    n = 0
    for word in jieba.cut(sentence):
        try:
            vec += model.wv[word]
            n += 1
        except KeyError:
            pass
    return vec if n==0 else vec/n

def load_data(w2v_model, f_name):

    # data = pd.read_excel(f_name)
    # data = data.astype(str)
    data = pd.read_excel(f_name, encoding='gb18030')
    
    temp_data = pd.DataFrame()
    for key in set(list(data['cluster_name'])):
        temp = data[data['cluster_name'] == key]
        if len(temp) >= 2000:
            temp = temp.sample(n=2000)
            temp_data = pd.concat([temp_data,temp])
            print(key,len(temp))
#        elif len(temp) < 100:
#            pass
        else:
            new_temp = temp
            while(len(new_temp)<2000):
                new_temp = pd.concat([new_temp,temp.sample(n=min(len(temp),2000-len(new_temp)))])
            temp_data = pd.concat([temp_data,new_temp])
            print(key,len(new_temp))
    data = temp_data
    
    label_map = {name: ind for ind, name in enumerate(data.cluster_name.unique())}
    name_map = {v: k for k, v in label_map.items()}
    print(name_map)
    vecs = np.array([get_vec(bl, w2v_model) for bl in data.text])
    labels = data.cluster_name.map(label_map)  # 从数据中提前列名
    print(labels[0:5],type(labels),333333333)
    # 数据切分
    x_train, x_test, y_train, y_test = train_test_split(vecs, labels, test_size=0.1, random_state=1)
    return x_train, x_test, y_train, y_test, name_map

def load_data2(w2v_model, f_name):

    # data = pd.read_excel(f_name)
    # data = data.astype(str)
    data = pd.read_excel(f_name, encoding='gb18030')
    label_map = {name: ind for ind, name in enumerate(data.cluster_name2.unique())}
    name_map = {v: k for k, v in label_map.items()}
    
    t_data=pd.DataFrame()
    v_data=pd.DataFrame()
    for v in name_map.values():
        temp = data[data['cluster_name2'] == v]
        t_data=pd.concat( [t_data,temp.iloc[0:int(0.1*len(temp)),:]] )
        v_data=pd.concat([v_data,temp.iloc[int(0.2*len(temp)):len(temp),:]])
    
    data = v_data
    data_test = t_data
    writer = pd.ExcelWriter('a.xlsx')
    data_test.to_excel(writer, sheet_name='sheet1')
    
    all_vec = []
    all_label = []
    expand = 3000
    for v in name_map.values():
        temp_data = data[data['cluster_name2'] == v]
        if len(temp_data) > expand:
            temp_data = temp_data.sample(n = expand)
            temp_vec = np.array([get_vec(bl, w2v_model) for bl in temp_data.text])
            temp_label = temp_data.cluster_name2.map(label_map)
        else:
            t_vec = np.array([get_vec(bl, w2v_model) for bl in temp_data.text])
            t_vec = Smote(t_vec,N=int(expand/len(temp_data)))
            temp_vec=t_vec.over_sampling()
            temp_label = temp_data.cluster_name2.map(label_map)
            while(len(temp_label))<expand:
                temp_label = temp_label.append(temp_label)
            temp_label = temp_label[0:len(temp_vec)]
            
        if len(all_vec):
            all_vec = np.concatenate([all_vec,temp_vec])
            all_label = np.concatenate([all_label,temp_label])
        else:
            all_vec = temp_vec
            all_label = temp_label
        print(v,':',len(temp_vec))
    vecs = all_vec
    labels = all_label
    
    print(name_map)
#    vecs = np.array([get_vec(bl, w2v_model) for bl in data.text])
#    labels = data.cluster_name.map(label_map)  # 从数据中提前列名

    # 数据切分
    x_train, x_test, y_train, y_test = train_test_split(vecs, labels, test_size=0.1, random_state=1)
    return x_train, x_test, y_train, y_test, name_map

def load_data3(w2v_model, f_name):
    smote_enn = SMOTEENN(random_state=0)
    data = pd.read_excel(f_name, encoding='gb18030')
    
#    temp_data = pd.DataFrame()
#    for key in set(list(data['cluster_name'])):
#        temp = data[data['cluster_name'] == key]
#        if len(temp) >= 2000:
#            temp = temp.sample(n=2000)
#            temp_data = pd.concat([temp_data,temp])
#            print(key,len(temp))
##        elif len(temp) < 100:
##            pass
#        else:
#            new_temp = temp
#            while(len(new_temp)<2000):
#                new_temp = pd.concat([new_temp,temp.sample(n=min(len(temp),2000-len(new_temp)))])
#            temp_data = pd.concat([temp_data,new_temp])
#            print(key,len(new_temp))
#    data = temp_data
    
    label_map = {name: ind for ind, name in enumerate(data.cluster_name.unique())}
    name_map = {v: k for k, v in label_map.items()}
    print(name_map)
    vecs = np.array([get_vec(bl, w2v_model) for bl in data.text])
    labels = data.cluster_name.map(label_map)  # 从数据中提前列名
    vecs,labels = smote_enn.fit_sample(vecs,labels)
    print(labels[0:5],type(labels),333333333)
    # 数据切分
    x_train, x_test, y_train, y_test = train_test_split(vecs, labels, test_size=0.1, random_state=1)
    return x_train, x_test, y_train, y_test, name_map

def models_metrics(models, x_test, y_test):
    for k, v in models.items():
        y_pred_list = v.predict(x_test, num_iteration=v.best_iteration)
        y_score = [max(x) for x in y_pred_list]
        y_pred=[list(x).index(max(x)) for x in y_pred_list]
    result = {k:{'f1-score-macro':f1_score(y_test, y_pred, average="macro"), 'precision':precision_score(y_test, y_pred, average="macro"),\
        'recall': recall_score(y_test,y_pred,average="macro"),'f1-score-micro':f1_score(y_test,y_pred, average="micro")} for k,v in models.items()}
    return result, y_pred, y_score


if __name__ == '__main__':
    import codecs
    result_path = r'D:\kst\work\libs\result.txt'
    save_name = r'D:\kst\work\libs\andro_intention_0327_v3.pkl'
    result_f = codecs.open(result_path, 'a', encoding='utf-8')

    w2v_f = r'D:\kst\work\models\word2vec专科模型\all_zhuanke.model'
#    w2v_f = r'D:\kst\work\models\fasttext专科100维\all_zhuanke.fast'
#    w2v_f = r'D:\kst\work\models\word2vec_size100_window5_mincount5'
    # w2v_f = '../word2vec/nanke_word2vec_v2.model'
    w2v_model = load_w2v(w2v_f)
    data_path = r'D:\git\专科数据标注\男科intention\nanke_intention_0326.xls'
#    data_path = r'D:\git\label_criterion\action标注规则\beauty_action_0321.xls'
#    data_path = r'D:\git\label_criterion\action标注规则\action标注数据汇总.xls'
    
    params = {
        "task": 'train',
        'boosting_type': 'gbdt',
        "objective": "multiclass",
        "num_leaves": 80,  #!!
        "num_class": 17,
        "learning_rate": 0.1,
        "max_depth": 7,  #!!
        'bagging_freq': 10,
        # 'subsample': 0.7,
        'shuffle': True,

        "lambda_l1": 0.6,  #!!
        "lambda_l2": 0.2,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,  #!!

    }

    x_train, x_test, y_train, y_test, name_map = load_data2(w2v_model, data_path)
    print(name_map)
    train_data = lgb.Dataset(x_train, label = y_train)
    validation_data = lgb.Dataset(x_test, label=y_test)
    gbm = lgb.train(params, train_data, num_boost_round=200, valid_sets=[validation_data], early_stopping_rounds=5)
    from sklearn.externals import joblib

    joblib.dump(gbm, save_name)


    clf_models = {
        'LightGBM': gbm
    }
    metrics_result, y_pred, y_score = models_metrics(clf_models, x_test, y_test)
    # raw_test['y_pred'] = y_pred
    # raw_test['score'] = y_score
    print(str(metrics_result))


    import re
    # target_names = ['无', '咨询项目', '咨询副作用', '咨询疗程', '描述症状', '咨询价格', '确定预约时间', '咨询地址', '咨询材料仪器治疗方法', '咨询联系方式', '咨询优惠活动', ]
    target_names = list(name_map.values())
    answers = gbm.predict(x_test)
    res = np.array([list(answers[i]).index(max(answers[i,:])) for i in range(0,len(answers))])
#    yt= list(map(lambda x: int(x), y_test))
#    res = list(map(lambda x: int(x), res))
    individual_metrics = classification_report(y_test, res, target_names=target_names)
    individual_metrics = [re.sub(pattern='\s{2,}', string=line, repl='::').split('::') for line in individual_metrics.split('\n') if line!='']
    individual_metrics = [line[1:] if len(line)==6 else line for line in individual_metrics]
    pd.DataFrame(individual_metrics[1:], columns=individual_metrics[0]).set_index('')

    result_f.write('训练时间：' + str(time.strftime('%Y.%m.%d.%H:%M:%S', time.localtime(time.time()))))
    result_f.write('\n参数列表\n')
    for p in params:
        result_f.write(str(p) + ':\t' + str(params[p]) + '\n')
    result_f.write('\n')
    for key, value in metrics_result['LightGBM'].items():
        result_f.write(str(key) + ':    ' + str(value) + '\n')
    result_f.write('-' * 100 + '\n\n')