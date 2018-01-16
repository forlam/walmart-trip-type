import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.getcwd()
os.chdir('/Users/saketguddeti/Desktop/git/walmart-trip-type')
np.set_printoptions(threshold = np.nan)

####################################################################################
####################################################################################
############################# Importing the data ###################################
####################################################################################
####################################################################################
data = pd.read_csv('Data/train.csv')
data.head()
data.shape

####################################################################################
####################################################################################
############################# Exploring the data ###################################
####################################################################################
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
'''
####################################################################################
####################################################################################
############################# Cleaning the dataset #################################
####################################################################################
####################################################################################

data = data.dropna(how = 'any') # Removing Null Value rows
data = data.loc[data['ScanCount'] > -1,:] # Removing rows for product return


# Aggregating number if items bought with respect to visit id and upc
data2 = data.groupby(['VisitNumber','Upc'])['ScanCount'].sum()
data2 = data2.reset_index()
data = data.drop('ScanCount',axis = 1)
data = pd.merge(left=data, right=data2, how='left', left_on=['VisitNumber','Upc'],right_on=['VisitNumber','Upc'])
data = data.drop_duplicates() # Removing duplicates in the data

unique_items = data.groupby(['VisitNumber', 'Upc']).size()
unique_items = unique_items.reset_index()
unique_items.columns = ('VisitNumber', 'Upc' , 'UniqueItems')
unique_items = unique_items.groupby('VisitNumber')['UniqueItems'].sum().reset_index()
total_items = data.groupby('VisitNumber')['ScanCount'].sum().reset_index()


unique_depts = data.groupby(['VisitNumber', 'DepartmentDescription']).size().reset_index()
unique_depts = unique_depts.groupby(['VisitNumber']).size().reset_index()
unique_depts.columns = ('VisitNumber', 'UniqueDepts')


unique_models = data.groupby(['VisitNumber', 'FinelineNumber']).size().reset_index()
unique_models = unique_models.groupby(['VisitNumber']).size().reset_index()
unique_models.columns = ('VisitNumber', 'UniqueModels')

data = pd.merge(left=data, right=unique_items, how = 'left', left_on ='VisitNumber',right_on='VisitNumber')
data = data.drop('ScanCount',axis=1)
data = pd.merge(left=data, right=total_items, how = 'left', left_on ='VisitNumber',right_on='VisitNumber')
data = pd.merge(left=data, right=unique_depts, how = 'left', left_on ='VisitNumber',right_on='VisitNumber')
data = pd.merge(left=data, right=unique_models, how = 'left', left_on ='VisitNumber',right_on='VisitNumber')

data2 = data

###########TFIDF################
idf_score = data2.groupby('Upc')['VisitNumber'].apply(list).reset_index()
total_visits = len(set(data2.VisitNumber))
idf_score['freq'] = idf_score['VisitNumber'].apply(lambda x: np.log(total_visits/len(x)))

idf_dic = {idf_score['Upc'][i]:idf_score['freq'][i] for i in range(idf_score.shape[0])}

dept_score = data[['DepartmentDescription','Upc']].drop_duplicates()
dept_score['dept_idf'] = dept_score['Upc'].map(idf_dic)
dept_score = dept_score.groupby('DepartmentDescription')['dept_idf'].agg('mean').reset_index()
dept_score = dept_score.set_index('DepartmentDescription').to_dict()['dept_idf']
data2['idf_score'] = data2['DepartmentDescription'].map(dept_score)

##################################

data2['FinelineNumber'] = data2['FinelineNumber'].apply(lambda x: str(x))

depts = data2.groupby('VisitNumber')['DepartmentDescription'].apply(list).reset_index()
fln = data2.groupby('VisitNumber')['FinelineNumber'].apply(list).reset_index()
idf = data2.groupby('VisitNumber')['idf_score'].agg('sum').reset_index()
data2 = data2.drop(['Upc','DepartmentDescription','FinelineNumber','idf_score'], axis = 1)
data2 = data2.drop_duplicates()
data2 = pd.merge(left=data2,right=depts,how='left',left_on='VisitNumber',right_on='VisitNumber')
data2 = pd.merge(left=data2,right=fln,how='left',left_on='VisitNumber',right_on='VisitNumber')
data2 = pd.merge(left=data2,right=idf,how='left',left_on='VisitNumber',right_on='VisitNumber')

####################################
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
train_desp_vec = vect.fit_transform(word_list)
train_desp_vec1 = vect1.fit_transform(word_list)

train_tfidf = np.hstack([train_desp_vec.toarray() , train_desp_vec1.toarray()])
from sklearn.decomposition import PCA
pca = PCA(n_components = 1000)
tfidf_data = pca.fit_transform(train_tfidf)
tfidf_data = pd.DataFrame(tfidf_data)
data2 = pd.concat([data2, tfidf_data], axis=1)

####################################################################################
####################################################################################
############################# Scaling the data #####################################
####################################################################################
####################################################################################

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
data2[['UniqueItems','ScanCount','UniqueDepts','UniqueModels','idf_score']] = sc_X.fit_transform(data2[['UniqueItems','ScanCount','UniqueDepts','UniqueModels','idf_score']])

####################################################################################
####################################################################################
############################# Encoding Categorical Data #####################################
####################################################################################
####################################################################################

# Encoding Categorical Data
x = pd.get_dummies(data2['Weekday'], drop_first = True)
data2 = pd.concat([data2,x], axis = 1)
data2 = data2.drop('Weekday', axis = 1)

from sklearn.preprocessing import MultiLabelBinarizer
multilabelbinarizer = MultiLabelBinarizer()
x = pd.DataFrame(multilabelbinarizer.fit_transform(data2['DepartmentDescription']), 
                 columns = multilabelbinarizer.classes_)
data2 = pd.concat([data2,x], axis = 1)
data2 = data2.drop('DepartmentDescription', axis = 1)


x = pd.DataFrame(multilabelbinarizer.fit_transform(data2['FinelineNumber']), 
                 columns = multilabelbinarizer.classes_)
data2 = data2.drop('FinelineNumber', axis = 1)

from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
pca_fln = pca.fit_transform(x)
pca_fln = pd.DataFrame(pca_fln, columns=["fln"+str(x) for x in range(100)])
data2 = pd.concat([data2,pca_fln], axis = 1)

######################################################

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
data2.iloc[:,-68:] = sc_X.fit_transform(data2.iloc[:,-68:])

'''
import networkx as nx    
G = nx.from_numpy_matrix(dept_corr.values)
G = nx.relabel_nodes(G, dict(enumerate(dept_corr.columns)))
nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_size = 4, width = 0.2, font_size = 3)
plt.savefig("labels.png", format="PNG",dpi=1000)
'''

####################################################

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data2.iloc[:,2:] ,data2.iloc[:,0],test_size = 0.2)

# Applying Random Forest Classifier
'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 250, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
sum(y_pred == y_test)/18041


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
sum(y_pred == y_test)/18041

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
sum(y_pred == y_test)/18041
'''









import xgboost as xgb

label_dict = pd.DataFrame(list(set(data.TripType)))
label_dict['ind'] = pd.Series(list(label_dict.index))
label_dict = label_dict.set_index(0).to_dict()['ind']
y_train = y_train.map(label_dict)
y_test = y_test.map(label_dict)

xg_train = xgb.DMatrix(X_train.values, label=y_train.values)
xg_test = xgb.DMatrix(X_test.values, label=y_test.values)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.2
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 38

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 15
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
y_pred = bst.predict(xg_test)
np.sum(y_pred == y_test.values) / y_test.shape[0]
