import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.getcwd()
os.chdir('/Users/saketguddeti/Desktop/git/walmart-trip-type')
np.set_printoptions(threshold = np.nan)


####################################################################################
############################# Importing the data ###################################
####################################################################################

train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

####################################################################################
############################# Exploring the data ###################################
####################################################################################
'''
data.info()
data.describe()

# TripType
unique_triptype = list(set(data.TripType))
triptype_freq = data.TripType.value_counts()
plt.hist(data.loc[data['TripType'] != 999,'TripType'])

# Weekday
weekday_freq = data.Weekday.value_counts()
data.groupby(['Weekday','TripType']).size()

len(set(data.DepartmentDescription))

test = data
test['leng'] = test['Upc'].apply(lambda x: len(str(x)))
set(test.leng)

# Extracting trips with only return items only
return_trips = data.groupby(['VisitNumber', 'ScanCount']).size().reset_index()
return_trips = return_trips.loc[return_trips['ScanCount'] < 0,:]
return_trips = pd.Series(list(set(return_trips['VisitNumber'])))
return_trips = return_trips.value_counts().reset_index()

return_trips2 = pd.merge(left=data,right=return_trips,how='inner',left_on='VisitNumber',right_on='index')
return_trips2 = return_trips2.groupby('VisitNumber').size().reset_index()
return_trips2 = return_trips2.loc[return_trips2[0] == 1,:]
return_trips3 = pd.merge(left = data.iloc[:,:2], right = return_trips2, how = 'inner',
               left_on = 'VisitNumber', right_on = 'VisitNumber')
return_trips3['TripType'] = 999
return_trips3 = return_trips3.drop(0, axis = 1)
return_trips3 = pd.merge(left = return_trips3, right = data.loc[data['type'] == 'Test',['TripType','VisitNumber']],
                         how = 'inner', left_on = 'VisitNumber', right_on = 'VisitNumber')
return_trips3 = return_trips3[['VisitNumber','TripType_x']].drop_duplicates()
'''
####################################################################################
############################# Cleaning the dataset #################################
####################################################################################

# Concatenating train and test data
train_data["type"] = "Train"
train_data = train_data.loc[pd.notnull(train_data['Upc']),:] 
test_data["type"] = "Test"
test_data.insert(0, column = 'TripType', value = [-1234 for i in range(test_data.shape[0])])
raw_data = pd.concat([train_data,test_data], axis = 0)

# Finding the most frequent UPC-Fineline Number for departments
most_freq_dept_upc_fln = raw_data.groupby(['DepartmentDescription','FinelineNumber','Upc']).size().reset_index()
most_freq_dept_upc_fln = most_freq_dept_upc_fln.sort_values(['DepartmentDescription',0], ascending = False)
most_freq_dept_upc_fln = most_freq_dept_upc_fln.groupby(['DepartmentDescription']).first().reset_index()
most_freq_dept_upc_fln.columns = ['DepartmentDescription','fln','up',0]

# Filling out missing UPC/FineLineNumber values using DepartmentDescription
data = pd.merge(left=raw_data, right=most_freq_dept_upc_fln,how='left',
                left_on='DepartmentDescription',right_on='DepartmentDescription')

#test = data[data.isnull().any(axis = 1)]
data['Upc'] = np.where((pd.isnull(data['Upc'])) & (pd.notnull(data['DepartmentDescription'])),
    data['up'], data['Upc'])
data['FinelineNumber'] = np.where((pd.isnull(data['FinelineNumber'])) & (pd.notnull(data['DepartmentDescription'])),
    data['fln'], data['FinelineNumber'])
data = data.drop(['up','fln',0],axis = 1)

# Filling out missing UPC/FinelIne/Department by the most popular ones wrt weekday
most_freq_week_dept = raw_data.groupby(['Weekday','DepartmentDescription','FinelineNumber','Upc']).size().reset_index()
most_freq_week_dept = most_freq_week_dept.sort_values(['Weekday',0], ascending = False)
most_freq_week_dept = most_freq_week_dept.groupby('Weekday').first().reset_index()
most_freq_week_dept.columns = ['Weekday','DeptDesc','fln','up', 0]

data = pd.merge(left=data, right=most_freq_week_dept, how='left',
                left_on = 'Weekday', right_on='Weekday')
data['DepartmentDescription'] = np.where(pd.isnull(data['DepartmentDescription']), 
         data['DeptDesc'], data['DepartmentDescription'])
data['Upc'] = np.where(pd.isnull(data['Upc']), 
         data['up'], data['Upc'])
data['FinelineNumber'] = np.where(pd.isnull(data['FinelineNumber']), 
         data['fln'], data['FinelineNumber'])
data = data.drop(['DeptDesc','up','fln',0],axis = 1)
# data = data.dropna(how = 'any') # Removing Null Value rows


# Removing entries with any product return and extracting visitnumbers with only return
return_trip_id = raw_data[['VisitNumber','type']].drop_duplicates()
data = data.loc[data['ScanCount'] > -1,:] 
return_trip_id = pd.merge(left = return_trip_id, right = data[['VisitNumber','type']].drop_duplicates(),
                         how = 'left', left_on = 'VisitNumber', right_on = 'VisitNumber')
return_trip_id = return_trip_id.loc[pd.isnull(return_trip_id['type_y']),:]
return_trip_id = return_trip_id.drop('type_y', axis = 1)
data2 = data
 
# Aggregating number of items bought with respect to visit number and UPC
upc_sum = data2.groupby(['VisitNumber','Upc'])['ScanCount'].sum()
upc_sum = upc_sum.reset_index()
data2 = data2.drop('ScanCount',axis = 1)
data2 = pd.merge(left=data2, right=upc_sum, how='left', left_on=['VisitNumber','Upc'],right_on=['VisitNumber','Upc'])
data2 = data2.drop_duplicates() # Removing duplicates in the data

# Aggregating number of unique items purchased per visit
unique_items = data2.groupby(['VisitNumber', 'Upc']).size().reset_index()
unique_items.columns = ('VisitNumber', 'Upc' , 'UniqueItems')
unique_items = unique_items.groupby('VisitNumber')['UniqueItems'].sum().reset_index()
total_items = data2.groupby('VisitNumber')['ScanCount'].sum().reset_index()

# Aggregating number of unique department items purchased per visit
unique_depts = data2.groupby(['VisitNumber', 'DepartmentDescription']).size().reset_index()
unique_depts = unique_depts.groupby(['VisitNumber']).size().reset_index()
unique_depts.columns = ('VisitNumber', 'UniqueDepts')

# Aggregating number of unique model items purchased per visit
unique_models = data2.groupby(['VisitNumber', 'FinelineNumber']).size().reset_index()
unique_models = unique_models.groupby(['VisitNumber']).size().reset_index()
unique_models.columns = ('VisitNumber', 'UniqueModels')

# Merging all the features into the dataframe
data2 = pd.merge(left=data2, right=unique_items, how = 'left', left_on ='VisitNumber',right_on='VisitNumber')
data2 = data2.drop('ScanCount',axis=1)
data2 = pd.merge(left=data2, right=total_items, how = 'left', left_on ='VisitNumber',right_on='VisitNumber')
data2 = pd.merge(left=data2, right=unique_depts, how = 'left', left_on ='VisitNumber',right_on='VisitNumber')
data2 = pd.merge(left=data2, right=unique_models, how = 'left', left_on ='VisitNumber',right_on='VisitNumber')


# Feature for reflecting the rarity of the item purchased and hence the department
idf_score = data2.groupby('Upc')['VisitNumber'].apply(list).reset_index()
total_visits = len(set(data2.VisitNumber))
idf_score['freq'] = idf_score['VisitNumber'].apply(lambda x: np.log(total_visits/len(x)))

idf_dic = {idf_score['Upc'][i]:idf_score['freq'][i] for i in range(idf_score.shape[0])}

dept_score = data2[['DepartmentDescription','Upc']].drop_duplicates()
dept_score['dept_idf'] = dept_score['Upc'].map(idf_dic)
dept_score = dept_score.groupby('DepartmentDescription')['dept_idf'].agg('mean').reset_index()
dept_score = dept_score.set_index('DepartmentDescription').to_dict()['dept_idf']
data2['idf_score'] = data2['DepartmentDescription'].map(dept_score)

# Aggregating Department, Fineline Number and IDF Score wrt Visit Number
data2['FinelineNumber'] = data2['FinelineNumber'].apply(lambda x: str(x))

depts = data2.groupby('VisitNumber')['DepartmentDescription'].apply(list).reset_index()
fln = data2.groupby('VisitNumber')['FinelineNumber'].apply(list).reset_index()
idf = data2.groupby('VisitNumber')['idf_score'].agg('sum').reset_index()
data2 = data2.drop(['Upc','DepartmentDescription','FinelineNumber','idf_score'], axis = 1)
data2 = data2.drop_duplicates()
data2 = pd.merge(left=data2,right=depts,how='left',left_on='VisitNumber',right_on='VisitNumber')
data2 = pd.merge(left=data2,right=fln,how='left',left_on='VisitNumber',right_on='VisitNumber')
data2 = pd.merge(left=data2,right=idf,how='left',left_on='VisitNumber',right_on='VisitNumber')

# Feature for TF-IDF scores for the department description
import nltk
from nltk.corpus import stopwords
import re
stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about',
             'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 
             'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 
             'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 
             'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these',
             'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 
             'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 
             'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any',
             'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
             'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 
             'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 
             'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 
             'it', 'how', 'further', 'was', 'here', 'than']

word_list = []
for lt in list(depts['DepartmentDescription']):
    try:
        lt = [x for x in lt if x not in stopwords and str(x) != 'nan']
        lt = re.sub('[^a-zA-Z]+', ' ', ' '.join(lt)).strip().lower().split()
        word_list.append(' '.join(lt))
    except:
        word_list.append(['None'])
        
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(ngram_range=(1,1), min_df=0, max_features = 1000)
vect1 = TfidfVectorizer(ngram_range=(2,9), min_df=0, analyzer = 'char', max_features = 2000)
vec = vect.fit_transform(word_list)
vec1 = vect1.fit_transform(word_list)

train_tfidf = np.hstack([vec.toarray() , vec1.toarray()])
from sklearn.decomposition import PCA
pca = PCA(n_components = 1000)
tfidf_data = pca.fit_transform(train_tfidf)
tfidf_data = pd.DataFrame(tfidf_data)
data2 = pd.concat([data2, tfidf_data], axis=1)

####################################################################################
############################# Encoding Categorical Data ############################
####################################################################################

# Encoding Weekday values
x = pd.get_dummies(data2['Weekday'], drop_first = True)
data2 = pd.concat([data2,x], axis = 1)
data2 = data2.drop('Weekday', axis = 1)

# Encoding Department values
from sklearn.preprocessing import MultiLabelBinarizer
multilabelbinarizer = MultiLabelBinarizer()
x = pd.DataFrame(multilabelbinarizer.fit_transform(data2['DepartmentDescription']), 
                 columns = multilabelbinarizer.classes_)
data2 = pd.concat([data2,x], axis = 1)
data2 = data2.drop('DepartmentDescription', axis = 1)

# Encoding Fineline Number values
x = pd.DataFrame(multilabelbinarizer.fit_transform(data2['FinelineNumber']), 
                 columns = multilabelbinarizer.classes_)
data2 = data2.drop('FinelineNumber', axis = 1)

from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
pca_fln = pca.fit_transform(x)
pca_fln = pd.DataFrame(pca_fln, columns=["fln"+str(x) for x in range(100)])
data2 = pd.concat([data2,pca_fln], axis = 1)

# Capturing interaction between highly associated departments
dept_corr = data.groupby(['VisitNumber','DepartmentDescription']).size().reset_index()
dept_corr = pd.merge(left=dept_corr,right=dept_corr,how='left',on='VisitNumber')
dept_corr = dept_corr.groupby(['DepartmentDescription_x','DepartmentDescription_y']).size().reset_index()
dept_corr = dept_corr.pivot_table(index='DepartmentDescription_x', columns='DepartmentDescription_y', values=0)
dept_corr = dept_corr.fillna(0)
for i in range(len(set(data.DepartmentDescription))):
    dept_corr.iloc[i,i] = 0
dept_corr = dept_corr.div(dept_corr.max(axis = 1), axis = 0)
dept_corr = dept_corr.values

dept_freq = data.groupby(['VisitNumber','DepartmentDescription']).size().reset_index()
dept_freq = dept_freq.pivot_table(index='VisitNumber', columns='DepartmentDescription', values=0)
dept_freq = dept_freq.fillna(0)
dept_freq_ind = dept_freq.index
dept_freq = dept_freq.values

dept_diff = dept_freq
for i in range(len(set(data.DepartmentDescription))):
    a = 0.2*dept_freq[:,i]
    a = np.reshape(a,(a.shape[0],1))
    b = dept_corr[i,:]
    c = b*a
    dept_diff = dept_diff + c

dept_diff = pd.DataFrame(dept_diff, columns = ["dept"+str(x) for x in range(len(set(data.DepartmentDescription)))])
dept_diff['VisitNumber'] = dept_freq_ind

data2 = pd.merge(left=data2,right=dept_diff,how='left',left_on='VisitNumber',right_on='VisitNumber')

'''
import networkx as nx    
G = nx.from_numpy_matrix(dept_corr.values)
G = nx.relabel_nodes(G, dict(enumerate(dept_corr.columns)))
nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_size = 4, width = 0.2, font_size = 3)
plt.savefig("labels.png", format="PNG",dpi=1000)
'''

# Scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
data2[['UniqueItems','ScanCount','UniqueDepts','UniqueModels','idf_score']] = sc_X.fit_transform(data2[['UniqueItems','ScanCount','UniqueDepts','UniqueModels','idf_score']])
data2.iloc[:,-68:] = sc_X.fit_transform(data2.iloc[:,-68:])


# Splitting the dataframe into train/test data
train = data2.loc[data2['type']=="Train",:]
X_train = train.iloc[:,3:]
y_train = train.iloc[:,0]

test = data2.loc[data2['type']=="Test",:]
X_test = test.iloc[:,3:]
y_test = test.iloc[:,0]

# Use XGBoost algorithm for multi-classification
import xgboost as xgb
label_dict = pd.DataFrame(list(set(train.TripType)))
label_dict['ind'] = pd.Series(list(label_dict.index))
label_dict = label_dict.set_index(0).to_dict()['ind']
y_train = y_train.map(label_dict)
# creating a dummy label set for test dataset
y_test = pd.Series([31 for i in range(X_test.shape[0])])

xg_train = xgb.DMatrix(X_train.values, label=y_train.values)
xg_test = xgb.DMatrix(X_test.values, label=y_test.values)
# setting up parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 38
watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 50
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
y_pred = bst.predict(xg_test)
y_pred = pd.DataFrame(y_pred, columns = ["TripType_"+str(i) for i in list(label_dict.keys())])
y_pred['VisitNumber'] = test['VisitNumber'].values
# concatenating return visits with TripType value '999'
y_pred2 = pd.DataFrame(return_trip_id.loc[return_trip_id['type_x'] == 'Test','VisitNumber'], columns = ['VisitNumber'])
y_pred2['TripType_999'] = 1
y_pred = pd.concat([y_pred, y_pred2], axis = 0)
y_pred = y_pred.fillna(0)

# Exporting final predictions to CSV file
y_pred.to_csv('prediction.csv', index = False)