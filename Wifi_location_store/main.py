import pandas as pd
import numpy as np
import pickle
%matplotlib inline
df_shop = pd.read_csv('../ccf_first_round_shop_info.csv')
df_user = pd.read_csv('../ccf_first_round_user_shop_behavior.csv')
df_test_data = pd.read_csv('../evaluation_public.csv',index_col='row_id')
df_user.index.rename('row_id',inplace =True)
df_user['time_stamp_d'] = df_user['time_stamp'].apply(lambda x: x.split(' ')[0])
df_user['time_stamp_h'] = df_user['time_stamp'].apply(lambda x: x.split(' ')[1])

#划分训练集和验证集（月末三天，包括周末一天，工作日2天）
vali_list = ['2017-08-30','2017-08-31']
df_vali_data = df_user[df_user['time_stamp_d'].apply(lambda x : x in vali_list)]
df_train_data = df_user[df_user['time_stamp_d'].apply(lambda x : x not in vali_list)]

mall_list = pd.read_csv('../others/mall_ID.csv',names=['mall_id'])
mall_list = list(mall_list['mall_id'])

df_train_data = pd.merge(df_train_data,df_shop,how='left',on = 'shop_id')
df_vali_data = pd.merge(df_vali_data,df_shop,how='left',on = 'shop_id')
df_train_data['longitude'] =df_train_data['longitude_x']
df_train_data['latitude'] =df_train_data['latitude_x']
df_vali_data['longitude'] =df_vali_data['longitude_x']
df_vali_data['latitude'] =df_vali_data['latitude_x']

df_train_new = df_train_data[df_test_data.columns]
df_train_new['shop_id'] = df_train_data['shop_id']
df_vali_new = df_vali_data[df_test_data.columns]
df_vali_new['shop_id'] = df_vali_data['shop_id']
df_train_new.index.rename('row_id',inplace =True)
df_vali_new.index.rename('row_id',inplace =True)

df_train_new.to_csv("../begain/df_train_data.csv")
df_vali_new.to_csv("../begain/df_vali_data.csv")
df_test_data.to_csv("../begain/df_test_data.csv")

wifiDict={
    'mall_id':[],
    'shop_id':[],
    'bssid':[],
    'strength':[],
    'connect':[]}

for index,row in df_train_new.iterrows():
    if  isinstance(row.wifi_infos,str):
        for wifi in row.wifi_infos.split(';'):
            info=wifi.split('|')
            wifiDict['mall_id'].append(row.mall_id)
            wifiDict['shop_id'].append(row.shop_id)
            wifiDict['bssid'].append(info[0])
            wifiDict['strength'].append(info[1])
            wifiDict['connect'].append(info[2])
df_wifi=pd.DataFrame(wifiDict)   
df_wifi.to_csv('../begain/df_wifi_train.csv')

wifiDictTest={
    'mall_id':[],
    'bssid':[],
    'strength':[],
    'connect':[]}
for index,row in df_test_data.iterrows():
    if  isinstance(row.wifi_infos,str):
        for wifi in row.wifi_infos.split(';'):
            info=wifi.split('|')
            wifiDictTest['mall_id'].append(row.mall_id)
            wifiDictTest['bssid'].append(info[0])
            wifiDictTest['strength'].append(info[1])
            wifiDictTest['connect'].append(info[2])
df_wifi_test=pd.DataFrame(wifiDictTest)   
df_wifi_test.to_csv('../begain/df_wifi_test.csv')

wifi_dict_value ={}
wifi_dict_count={}
for mall_id_ in mall_list :
    print mall_id_
    df_mall_all = df_wifi[df_wifi['mall_id']==mall_id_]
    test_list = list(df_wifi_test[df_wifi_test['mall_id']==mall_id_]['bssid'].unique())
    df_mall = df_mall_all[df_mall_all['bssid'].apply(lambda x :x in test_list)]
    df_mall['strength'] = df_mall['strength'].astype(int)
    df_table_wifi = pd.pivot_table(df_mall,index=['shop_id'],columns =['bssid'],
                                         values=['strength'],aggfunc={'strength':np.mean})
    wifi_vector = pd.DataFrame(index=df_table_wifi.index,columns=list(df_table_wifi.columns.levels[1]),data=df_table_wifi.values) 
    shop_list = list(df_shop[df_shop['mall_id'] == mall_id_]['shop_id'])   
    list_not_in= list(set(shop_list).difference(set(list( wifi_vector.index))))
    df_not_in = pd.DataFrame(index=list_not_in,columns=list(wifi_vector.columns))
    wifi_vector = wifi_vector.append(df_not_in)   
    wifi_dict_value[mall_id_] = wifi_vector
    
    df_table_wifi2 = pd.pivot_table(df_mall,index=['shop_id'],columns =['bssid'],
                                         values=['strength'],aggfunc={'strength':len})
    wifi_vector2 = pd.DataFrame(index=df_table_wifi2.index,columns=list(df_table_wifi2.columns.levels[1]),data=df_table_wifi2.values)  
    list_not_in2= list(set(shop_list).difference(set(list( wifi_vector2.index))))
    df_not_in2 = pd.DataFrame(index=list_not_in2,columns=list(wifi_vector2.columns))
    wifi_vector2 = wifi_vector2.append(df_not_in2)   
    wifi_dict_count[mall_id_] = wifi_vector2
    
    
f7 = open("../begain/wifi_value_dict.txt","wb")
pickle.dump(wifi_dict_value,f7)
f7.close()
f8 = open("../begain/wifi_count_dict.txt","wb")
pickle.dump(wifi_dict_count,f8)
f8.close()



wifi500_dict_value ={}
wifi500_dict_count={}
for mall_id_ in mall_list :
    print mall_id_
    df_mall = df_wifi_in_test_dict[mall_id_]
    most500list = df_mall['bssid'].value_counts().index[0:500]
    df_mall = df_mall[df_mall['bssid'].apply(lambda x: x in most500list)]
    df_table_wifi = pd.pivot_table(df_mall,index=['shop_id'],columns =['bssid'],
                                         values=['strength'],aggfunc={'strength':np.mean})
    wifi_vector = pd.DataFrame(index=df_table_wifi.index,columns=list(df_table_wifi.columns.levels[1]),data=df_table_wifi.values) 
    shop_list = list(df_shop[df_shop['mall_id'] == mall_id_]['shop_id'])   
    list_not_in= list(set(shop_list).difference(set(list( wifi_vector.index))))
    df_not_in = pd.DataFrame(index=list_not_in,columns=list(wifi_vector.columns))
    wifi_vector = wifi_vector.append(df_not_in)   
    wifi500_dict_value[mall_id_] = wifi_vector
    
    df_table_wifi2 = pd.pivot_table(df_mall,index=['shop_id'],columns =['bssid'],
                                         values=['strength'],aggfunc={'strength':len})
    wifi_vector2 = pd.DataFrame(index=df_table_wifi2.index,columns=list(df_table_wifi2.columns.levels[1]),data=df_table_wifi2.values)  
    list_not_in2= list(set(shop_list).difference(set(list( wifi_vector2.index))))
    df_not_in2 = pd.DataFrame(index=list_not_in2,columns=list(wifi_vector2.columns))
    wifi_vector2 = wifi_vector2.append(df_not_in2)   
    wifi500_dict_count[mall_id_] = wifi_vector2
    
f7 = open("../begain/wifi500_value_dict.txt","wb")
pickle.dump(wifi500_dict_value,f7)
f7.close()
f8 = open("../begain/wifi500_count_dict.txt","wb")
pickle.dump(wifi500_dict_count,f8)
f8.close()



df_wifi_in_test_dict={}
for mall_id_ in mall_list :
    df_mall_all = df_wifi[df_wifi['mall_id']==mall_id_]
    test_list = list(df_wifi_test[df_wifi_test['mall_id']==mall_id_]['bssid'].unique())
    df_mall = df_mall_all[df_mall_all['bssid'].apply(lambda x :x in test_list)]
    df_mall['strength'] = df_mall['strength'].astype(int)
    df_wifi_in_test_dict[mall_id_]=df_mall
f8 = open("../begain/df_wifi_in_test_dict.txt","wb")
pickle.dump(df_wifi_in_test_dict,f8)
f8.close()



# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
import heapq
import bottleneck
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
%matplotlib inline

mall_list = pd.read_csv('../others/mall_ID.csv',names=['mall_id'])
mall_list = list(mall_list['mall_id'])
df_train_data = pd.read_csv("../data/data/df_train_data.csv")
df_vali_data= pd.read_csv("../data/data/df_vali_data.csv")
df_test_data = pd.read_csv("../data/data/df_test_data.csv")
df_shop = pd.read_csv('../ccf_first_round_shop_info.csv')
f5 = open("../mydata/shop_label_dict.txt","rb")
shop_label_dict = pickle.load(f5)
f5.close()
f6 = open("../mydata/label_shop_dict.txt","rb")
label_shop_dict = pickle.load(f6)
f6.close()

f2 = open("../begain/wifi0_count_dict.txt,"rb")
wifi_vector_dict = pickle.load(f2)
f2.close()
f3 = open("../data/1024/private_wifi_55.txt","rb")
private_wifi_55 = pickle.load(f3)
f3.close()
f4 = open("../data/1024/max_wifi_dict.txt","rb")
max_wifi_dict = pickle.load(f4)
f4.close()
f5 = open("../data/1024/true_wifi_all.txt","rb")
true_wifi_dict = pickle.load(f5)
f5.close()
f7 = open("../data/data/shop_time_dict2.txt","rb")
shop_time_dict = pickle.load(f7)
f7.close()
f8 = open("../data/data/shop_user_dict2.txt","rb")
shop_user_dict = pickle.load(f8)
f8.close()
f9 = open("../data/data/user_price_dict2.txt","rb")
user_price_dict = pickle.load(f9)
f9.close()

vali_label_list = []
vali_predict_list = []
forest_acc_dict ={}
#mall_list2 = ['m_1263']#,'m_1175'
#mall_list2 = ['m_2415']#m_1175'
#加距离
mall_list2 = ['m_1175']
for mall_id_ in mall_list:
    df_train_m = df_train_data[df_train_data['mall_id'] == mall_id_]
    df_train_m['time_stamp'] = pd.to_datetime(pd.Series(df_train_m['time_stamp']))
    df_train_m['time'] = df_train_m['time_stamp'].dt.dayofweek.astype('str') + df_train_m['time_stamp'].dt.hour.astype('str')
    df_train_m = df_train_m.set_index('row_id')
    df_vali_m = df_vali_data[df_vali_data['mall_id'] == mall_id_]
    df_vali_m['time_stamp'] = pd.to_datetime(pd.Series(df_vali_m['time_stamp']))
    df_vali_m['time'] = df_vali_m['time_stamp'].dt.dayofweek.astype('str') + df_vali_m['time_stamp'].dt.hour.astype('str')
    df_vali_m = df_vali_m.set_index('row_id')
    df_test_m = df_test_data[df_test_data['mall_id'] == mall_id_]
    df_test_m['time_stamp'] = pd.to_datetime(pd.Series(df_test_m['time_stamp']))
    df_test_m['time'] = df_test_m['time_stamp'].dt.dayofweek.astype('str') + df_test_m['time_stamp'].dt.hour.astype('str')
    df_test_m = df_test_m.set_index('row_id')
    test_pre_path  ='../data/1027/all/test_'+ mall_id_ +'.csv'
    wifi_vector = wifi_vector_dict[mall_id_]    
    wifi_true_vector = true_wifi_dict[mall_id_]
    wifi_max_vector = max_wifi_dict[mall_id_]
    wifi_private_vector = private_wifi_55[mall_id_]
    user_price = user_price_dict[mall_id_]
#    wifi_private_vector = wifi_max_vector
    shop_time_vector = shop_time_dict[mall_id_]
    shop_user_vector = shop_user_dict[mall_id_]
    shop_all = df_shop[df_shop ['mall_id']==mall_id_]
    shop_list = list(shop_all['shop_id'])
    
    df_train_list = df_train_m['wifi_infos'].apply(map_wifi_vector)
    df_train_new = df_train_m.join(df_train_list,how='left',rsuffix='_wifi')
    
    df_train_list2 = df_train_m['wifi_infos'].apply(map_wifi_true)
    df_train_new = df_train_new.join(df_train_list2,how='left',rsuffix='_true')
    
    df_train_list3 = df_train_m['wifi_infos'].apply(map_wifi_max)
    df_train_new = df_train_new.join(df_train_list3,how='left',rsuffix='_max')
    long1 = df_train_m['longitude'].mean()
    lat1 = df_train_m['latitude'].mean()
    df_train_new['distance'] = df_train_new.apply(map_distance,axis=1)

    df_train_private_wifi_1 = df_train_m['wifi_infos'].apply(map_wifi_max)
    df_train_private_wifi_1.fillna(0,inplace =True)
    df_train_private_wifi_1.index.names=['row_id']
    df_train_shop_time_2 = df_train_m['time'].apply(map_shop_time)
    df_train_shop_time_2.fillna(0,inplace =True)
    df_train_shop_time_2.index.names=['row_id']
    df_train_shop_user_3 = df_train_m['user_id'].apply(map_shop_user)
    df_train_shop_user_3.fillna(0,inplace =True)
    df_train_shop_user_3.index.names=['row_id']
    
    df_train_new = df_train_new.join(df_train_private_wifi_1,how='left',rsuffix='_pri_wifi')
    df_train_new = df_train_new.join(df_train_shop_time_2,how='left',rsuffix='_shop_time')
    df_train_new = df_train_new.join(df_train_shop_user_3,how='left',rsuffix='_shop_user')
    df_train_new['price'] = df_train_new['user_id'].apply(map_user_price)
    
    df_vali_list = df_vali_m['wifi_infos'].apply(map_wifi_vector)
    df_vali_new = df_vali_m.join(df_vali_list,how='left',rsuffix='_wifi')   
    df_vali_list2 = df_vali_m['wifi_infos'].apply(map_wifi_true)
    df_vali_new = df_vali_new.join(df_vali_list2,how='left',rsuffix='_true')    
    df_vali_list3 = df_vali_m['wifi_infos'].apply(map_wifi_max)
    df_vali_new = df_vali_new.join(df_vali_list3,how='left',rsuffix='_max')
    df_vali_new['distance'] = df_vali_new.apply(map_distance,axis=1)
    
    df_vali_private_wifi_1 = df_vali_m['wifi_infos'].apply(map_wifi_max)
    df_vali_private_wifi_1.fillna(0,inplace =True)
    df_vali_private_wifi_1.index.names=['row_id']
    df_vali_shop_time_2 = df_vali_m['time'].apply(map_shop_time)
    df_vali_shop_time_2.fillna(0,inplace =True)
    df_vali_shop_time_2.index.names=['row_id']
    df_vali_shop_user_3 = df_vali_m['user_id'].apply(map_shop_user)
    df_vali_shop_user_3.fillna(0,inplace =True)
    df_vali_shop_user_3.index.names=['row_id']
    
    df_vali_new = df_vali_new.join(df_vali_private_wifi_1,how='left',rsuffix='_pri_wifi')
    df_vali_new = df_vali_new.join(df_vali_shop_time_2,how='left',rsuffix='_shop_time')
    df_vali_new = df_vali_new.join(df_vali_shop_user_3,how='left',rsuffix='_shop_user')
    df_vali_new['price'] = df_vali_new['user_id'].apply(map_user_price)
    
    df_test_list = df_test_m['wifi_infos'].apply(map_wifi_vector)
    df_test_new = df_test_m.join(df_test_list,how='left',rsuffix='_wifi')
    df_test_list2 = df_test_m['wifi_infos'].apply(map_wifi_true)
    df_test_new = df_test_new.join(df_test_list2,how='left',rsuffix='_true')
    df_test_list3 = df_test_m['wifi_infos'].apply(map_wifi_max)
    df_test_new = df_test_new.join(df_test_list3,how='left',rsuffix='_max')
    df_test_new['distance'] = df_test_new.apply(map_distance,axis=1)
    
    df_test_private_wifi_1 = df_test_m['wifi_infos'].apply(map_wifi_max)
    df_test_private_wifi_1.fillna(0,inplace =True)
    df_test_private_wifi_1.index.names=['row_id']
    df_test_shop_time_2 = df_test_m['time'].apply(map_shop_time)
    df_test_shop_time_2.fillna(0,inplace =True)
    df_test_shop_time_2.index.names=['row_id']
    df_test_shop_user_3 = df_test_m['user_id'].apply(map_shop_user)
    df_test_shop_user_3.fillna(0,inplace =True)
    df_test_shop_user_3.index.names=['row_id']
    df_test_new['price'] = df_test_new['user_id'].apply(map_user_price)
    
    df_test_new = df_test_new.join(df_test_private_wifi_1,how='left',rsuffix='_pri_wifi')
    df_test_new = df_test_new.join(df_test_shop_time_2,how='left',rsuffix='_shop_time')
    df_test_new = df_test_new.join(df_test_shop_user_3,how='left',rsuffix='_shop_user')
    
    train_data_row = df_train_new
    train_data_row['label']=train_data_row['shop_id']
#    train_drop_list = list(train_data_row.columns)[0:10]
    train_drop_list =[ 'user_id', 'shop_id', 'time_stamp', 'wifi_infos','mall_id']
    train_data_drop = train_data_row.drop(train_drop_list, axis=1)
    train_data_drop['label']= train_data_drop['label'].apply(lambda x :shop_label_dict[mall_id_][x])
    
    vali_data_row = df_vali_new
    vali_data_row['label']= vali_data_row['shop_id']
#    vali_drop_list = list(vali_data_row.columns)[0:10]
    vali_drop_list =['user_id', 'shop_id', 'time_stamp', 'wifi_infos','mall_id']
    vali_data_drop = vali_data_row.drop(vali_drop_list, axis=1)
    vali_data_drop['label']= vali_data_drop['label'].apply(lambda x :shop_label_dict[mall_id_][x])
    
    test_data_row = df_test_new
#    test_drop_list = list(test_data_row.columns)[0:9]
    test_drop_list =['Unnamed: 0', 'user_id', 'time_stamp', 'wifi_infos','mall_id']
    test_data_drop = test_data_row.drop(test_drop_list, axis=1)

 #   train_data_drop = train_data_drop.append(vali_data_drop)
    
    np_train = np.array(train_data_drop.fillna(value=0).values) 
    X_train = np_train[:,:-1] 
    y_train = np_train[:,-1].astype(int)
#    X_train = X_train[:,111:]
        
    np_vali = np.array(vali_data_drop.fillna(value=0).values) 
    X_vali = np_vali[:,:-1] 
    y_vali = np_vali[:,-1].astype(int)

#    X_vali = X_vali[:,111:]
    np_test = np.array(test_data_drop.fillna(value=0).values) 
    X_test = np_test
    
    columns_list = list(np.sort(train_data_drop['label'].unique()))
    print "The model is RandomForest"
    clf = RandomForestClassifier(n_estimators =80, min_samples_split=3,random_state=10)
#max_features=10,max_depth=1, max_features='sqrt'
    
    tree_model = clf.fit(X_train, y_train)
 
########    importances = clf.feature_importances_
#    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
#    indices = np.argsort(importances)[::-1]
# Print the feature ranking
#    print("Feature ranking:")

#    for f in range(X_train.shape[1]):
#        print("%s. feature %d (%f)" % (test_data_drop.columns[indices[f]], indices[f], importances[indices[f]]))
    y_pred = clf.predict(X_vali)
    predictions = [round(value) for value in y_pred]
    vali_label_list.append(y_vali)
    vali_predict_list.append(y_pred)
    accuracy = accuracy_score(y_vali, predictions)
    print mall_id_
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    forest_acc_dict[mall_id_] = accuracy
    
    test_pre = clf.predict_proba(X_test)
    label_np = np.array(test_pre)
    
    columns_list = list(np.sort(train_data_drop['label'].unique()))
    test_pre_df = pd.DataFrame(index = test_data_row.index, columns=columns_list,data = label_np)
f6 = open("../data/1027/all/forest_acc_dict.txt","wb")
pickle.dump(forest_acc_dict,f6)
f6.close()

def map_wifi_true(x):
    a = 0
    wifi_dict = {}
    wifi_dict2 ={}
    wifi_list =[]
    for wifi in x.split(';'):
        if a <10:
            info=wifi.split('|')
            if info[0] in list(wifi_true_vector.columns):
                wifi_dict[info[0]]= info[1]
                wifi_dict2[info[1]]=info[0]
                a = a + 1            
    wifi_list = [wifi_dict2[v] for v in sorted(wifi_dict.values())]
#    print a
    if len(wifi_list)>0 :
#        print wifi_list[0]
        return wifi_true_vector[wifi_list[0]]
    else:
        return pd.Series(index =wifi_true_vector.index,data = 0)
def map_wifi_private(x):
    a = 0
    wifi_dict = {}
    wifi_dict2 ={}
    wifi_list =[]
    for wifi in x.split(';'):
        if a <10:
            info=wifi.split('|')
            if info[0] in list( wifi_private_vector.columns):
                wifi_dict[info[0]]= info[1]
                wifi_dict2[info[1]]=info[0]
                a = a + 1            
    wifi_list = [wifi_dict2[v] for v in sorted(wifi_private_vector.values())]
#    print a
    if len(wifi_list)>0 :
#        print wifi_list[0]
        return  wifi_private_vector[wifi_list[0]]
    else:
        return pd.Series(index= wifi_private_vector.index,data = 0)
def map_shop_time(x):
    if x in shop_time_vector.columns:
        return shop_time_vector[x]
    else:
        return pd.Series(index = shop_time_vector.index,data =0)
def map_shop_user(x):
    if x in shop_user_vector.columns:
        return shop_user_vector[x]
    else:
        return pd.Series(index = shop_user_vector.index,data =0)
def map_user_price(x):
     if x in user_price.keys():
        return user_price[x]
     else:
        return 0

def map_wifi_vector(x):
#    wifi_v =np.zeros(wifi_vector.shape[1])
    wifi_v = (wifi_vector.shape[1])*[0]
    for wifi in x.split(';'):
        info=wifi.split('|')
        if info[0]in list(wifi_vector.columns):
            if float(info[1])> -20:
                strg = -20
            else:
                if float(info[1])< -90:
                    strg = -90
                else :
                    strg = float(info[1])               
            wifi_v[list(wifi_vector.columns).index(info[0])] = 10**((0.016*(strg)+2.5))
    return pd.Series(index = wifi_vector.columns, data = wifi_v) 

def map_wifi_max(x):
    a = 0
    wifi_dict = {}
    wifi_dict2 ={}
    wifi_list =[]
    for wifi in x.split(';'):
        if a <10:
            info=wifi.split('|')
            if info[0] in list(wifi_max_vector.columns):
                wifi_dict[info[0]]= info[1]
                wifi_dict2[info[1]]=info[0]
                a = a + 1            
    wifi_list = [wifi_dict2[v] for v in sorted(wifi_dict.values())]
    if len(wifi_list)>1 :
  #      print wifi_dict[wifi_list[0]],wifi_dict[wifi_list[1]]
        return wifi_max_vector[wifi_list[0]]
    else:
        return pd.Series(index =wifi_max_vector.index,data =0)

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = (lat2 - lat1)*3
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 
    return c * r * 1000
#latitude	longitude
#longitude	latitude	
def map_distance(df):
    return haversine(long1,lat1,df['longitude'],df['latitude'])

def merge_result(df,df_wifi_1,df_shop_time_2,df_shop_user_3):
    df_result = pd.DataFrame(index = df.index,columns = shop_list)
    for x in shop_list:
        if (x in df.columns) and (x in df_wifi_1.columns):
            df_result[x] = df[x]*0.3 + df_wifi_1[x]*0.3 + df_shop_time_2[x]*0.15 + df_shop_user_3[x]*0.25
 #           print "1"
        else:
            if x in df.columns:
                df_result[x] = df[x]*0.5+ df_shop_time_2[x]*0.2 + df_shop_user_3[x]*0.3
 #               print "2"
            elif x in df_wifi_1.columns:
                df_result[x] = df_wifi_1[x]*0.35 + df_shop_time_2[x]*0.25 + df_shop_user_3[x]*0.4
               
            else:
                df_result[x] =  df_shop_time_2[x]*0.25 + df_shop_user_3[x]*0.75

    
    return df_result




shop_time_dict={}
shop_user_dict={}
for mall_id_ in mall_list:    
    print mall_id_
    df_mall = df_train_data[df_train_data['mall_id'] == mall_id_]
    #df_mall['max_wifi']=df_mall['wifi_infos'].apply(map_wifi_max)
 
    shop_list = list(df_shop[df_shop['mall_id'] == mall_id_]['shop_id'])   
    df_table_user_count = pd.pivot_table(df_mall,index=['shop_id'],columns =['user_id'],
                                         values=['wifi_infos'],aggfunc={'wifi_infos':len})
    shop_user_vector = pd.DataFrame(index=df_table_user_count.index,columns=list(df_table_user_count.columns.levels[1]),data=df_table_user_count.values)
    shop_user_vector = shop_user_vector.apply(lambda x: x*1.0/np.sum(x)) 
      
    list_not_in= list(set(shop_list).difference(set(list( shop_user_vector.index))))
    df_not_in = pd.DataFrame(index=list_not_in,columns=list(shop_user_vector.columns))
    shop_user_vector =  shop_user_vector.append(df_not_in)    
    shop_user_vector=shop_user_vector.fillna(0)
    shop_user_dict[mall_id_] =  shop_user_vector
    
    df_table_time_count = pd.pivot_table(df_mall,index=['shop_id'],columns =['time'],
                                         values=['wifi_infos'],aggfunc={'wifi_infos':len})
    shop_time_vector = pd.DataFrame(index=df_table_time_count.index,columns=list(df_table_time_count.columns.levels[1]),data=df_table_time_count.values)
    shop_time_vector = shop_time_vector.apply(lambda x: x*1.0/np.sum(x)) 
    
    
    list_not_in= list(set(shop_list).difference(set(list( shop_time_vector.index))))
    df_not_in = pd.DataFrame(index=list_not_in,columns=list(shop_time_vector.columns))
    shop_time_vector =  shop_time_vector.append(df_not_in)    
    shop_time_vector=shop_time_vector.fillna(0)
    shop_time_dict[mall_id_] =  shop_time_vector
f7 = open("../data/data/shop_time_dict2.txt","wb")
pickle.dump(shop_time_dict,f7)
f7.close()    
f8 = open("../data/data/shop_user_dict2.txt","wb")
pickle.dump(shop_user_dict,f8)
f8.close()   

df_all = pd.read_csv('../data/1023/df_train_wifi_in_test_all.csv')
private_wifi_all ={}
private_wifi_70 = {}
private_wifi_55 = {}
for mall_id_ in mall_list2 :
    print mall_id_
    p_all ={}
    p_70 = {}
    p_55 = {}
    df_mall = df_all[df_all['mall_id'] == mall_id_]
    df_mall_true = df_mall[df_mall['connect']==True] 
    df_table_wifi_count = pd.pivot_table(df_mall_true,index=['shop_id'],columns =['bssid'],
                                         values=['strength'],aggfunc={'strength':len})
    shop_wifi_vector = pd.DataFrame(index=df_table_wifi_count.index,columns=list(df_table_wifi_count.columns.levels[1]),data=df_table_wifi_count.values)
    shop_wifi_vector = shop_wifi_vector.apply(lambda x: x*1.0/np.sum(x))
    df_table_wifi_value = pd.pivot_table(df_mall_true,index=['shop_id'],columns =['bssid'],
                                         values=['strength'],aggfunc={'strength':np.mean})
    shop_wifi_v_vector = pd.DataFrame(index=df_table_wifi_value.index,columns=list(df_table_wifi_value.columns.levels[1]),data=df_table_wifi_value.values)
    for s,w in shop_wifi_vector.iterrows():
        for y in shop_wifi_vector.columns:
            if w[y] == 1:
                p_all[y] = s
                if shop_wifi_v_vector.loc[s,y]>-70 :
                    p_70[y] = s
                if shop_wifi_v_vector.loc[s,y]>-55 :
                    p_55[y] = s
    
    private_wifi_all[mall_id_] = p_all
    private_wifi_70 [mall_id_] = p_70
    private_wifi_55 [mall_id_] = p_55
f7 = open("../data/1027/private_wifi_all.txt","wb")
pickle.dump(private_wifi_all,f7)
f7.close()
f8 = open("../data/1027/private_wifi_70.txt","wb")
pickle.dump(private_wifi_70,f8)
f8.close()
f9 = open("../data/1027/private_wifi_55.txt","wb")
pickle.dump(private_wifi_55,f9)
f9.close()


true_wifi_all ={}
for mall_id_ in mall_list :
    print mall_id_
    p_all ={}
    df_mall = df_all[df_all['mall_id'] == mall_id_]
    df_mall_true = df_mall[df_mall['connect']==True] 
    df_table_wifi_count = pd.pivot_table(df_mall_true,index=['shop_id'],columns =['bssid'],
                                         values=['strength'],aggfunc={'strength':len})
    shop_wifi_vector = pd.DataFrame(index=df_table_wifi_count.index,columns=list(df_table_wifi_count.columns.levels[1]),data=df_table_wifi_count.values)
    shop_wifi_vector = shop_wifi_vector.apply(lambda x: x*1.0/np.sum(x))   
    true_wifi_all[mall_id_] =  shop_wifi_vector
f7 = open("../data/1024/true_wifi_all.txt","wb")
pickle.dump(true_wifi_all,f7)
f7.close()

def map_wifi_max(x):
    a = 0
    wifi_dict = {}
    wifi_dict2 ={}
    wifi_list =[]
    for wifi in x.split(';'):
        if a <10:
            info=wifi.split('|')
            if info[0] in list(wifi_vector.columns):
                wifi_dict[info[0]]= info[1]
                wifi_dict2[info[1]]=info[0]
                a = a + 1            
    wifi_list = [wifi_dict2[v] for v in sorted(wifi_dict.values())]
    if len(wifi_list)>0 :
        return wifi_list[0]

max_wifi_dict={}

for mall_id_ in mall_list:
    wifi_vector = wifi_vector_dict[mall_id_]    
    print mall_id_
    df_mall = df_train_data[df_train_data['mall_id'] == mall_id_]
    df_mall['max_wifi']=df_mall['wifi_infos'].apply(map_wifi_max)
    df_table_wifi_count = pd.pivot_table(df_mall,index=['shop_id'],columns =['max_wifi'],
                                         values=['user_id'],aggfunc={'user_id':len})
    shop_wifi_vector = pd.DataFrame(index=df_table_wifi_count.index,columns=list(df_table_wifi_count.columns.levels[1]),data=df_table_wifi_count.values)
    shop_wifi_vector = shop_wifi_vector.apply(lambda x: x*1.0/np.sum(x))   
    max_wifi_dict[mall_id_] =  shop_wifi_vector
f7 = open("../data/1024/max_wifi_dict.txt","wb")
pickle.dump(max_wifi_dict,f7)
f7.close()    


 forest = RandomForestClassifier(n_estimators =200 ,max_features='sqrt',random_state=10)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    X_train = X_train[:,indices[0:500]]
    X_vali = X_vali[:,indices[0:500]]
    X_test = X_test[:,indices[0:500]]


user_price_dict={}
mall_list2 = ['m_1263']
for mall_id_ in mall_list:    
    print mall_id_
    df_mall = df_train_data[df_train_data['mall_id_x'] == mall_id_]
    shop_list = list(df_shop[df_shop['mall_id'] == mall_id_]['shop_id'])   
    df_table_user_count = pd.pivot_table(df_mall,index=['user_id'],
                                         values=['price'],aggfunc={'price':np.mean})
    shop_user_vector =dict(df_table_user_count) 
    user_price_dict[mall_id_] = shop_user_vector
f7 = open("../data/data/user_price_dict2.txt","wb")
pickle.dump(user_price_dict,f7)
f7.close()    

diff_list =[]
for x in range(len(vali_label_list)):
    for y in range(len(vali_label_list[x])):
        diff_list.append(vali_label_list[x][y]-vali_predict_list[x][y])
num = 0
for x in diff_list:
#    print type(x)
    if x > 0 or x < 0 :
        num = num +1
print num
print len(diff_list)
print 1 - num*1.0/len(diff_list)

mall_0 = mall_list[0]
mall_other = mall_list[1:]

df_pre = pd.read_csv('../data/1026/fusion2/test_'+ mall_0 +'.csv',index_col=['row_id'])
df_T = df_pre.T
result_list = []
for x in df_T.columns :
    result_list.append(df_T[x].argmax())
df_p = pd.DataFrame(index = df_pre.index, columns=['shop_id'],data = result_list )
#df_p['shop_id'] = df_p['shop_id'].apply(lambda x :label_shop_dict[mall_0][int(x)])
df_sub = df_p


#mall_list = mall_list[1:]
for mall_id_ in mall_other:
    test_pre_path = '../data/1026/fusion2/test_'+ mall_id_ +'.csv'
    df_pre = pd.read_csv(test_pre_path,index_col=['row_id'])
    df_T = df_pre.T
    result_list = []
    for x in df_T.columns :
        result_list.append(df_T[x].argmax())
    df_p = pd.DataFrame(index = df_pre.index, columns=['shop_id'],data = result_list )
#    df_p['shop_id'] = df_p['shop_id'].apply(lambda x :label_shop_dict[mall_id_][int(x)])
    df_sub = df_sub.append(df_p)
df_sub.to_csv('../submit/1026/row333.csv')












