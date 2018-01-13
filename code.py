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

data2['FinelineNumber'] = data2['FinelineNumber'].apply(lambda x: str(x))
data2 = data2.drop(['Upc'], axis=1)

depts = data2.groupby('VisitNumber')['DepartmentDescription'].apply(list).reset_index()
data2 = data2.drop('DepartmentDescription', axis = 1)
data2 = data2.drop_duplicates()
data2 = pd.merge(left=data2,right=depts,how='left',left_on='VisitNumber',right_on='VisitNumber')

####################################################################################
####################################################################################
############################# Scaling the data #####################################
####################################################################################
####################################################################################

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
data2[['UniqueItems','ScanCount','UniqueDepts','UniqueModels']] = sc_X.fit_transform(data2[['UniqueItems','ScanCount','UniqueDepts','UniqueModels']])

####################################################################################
####################################################################################
############################# Encoding Categorical Data #####################################
####################################################################################
####################################################################################

from sklearn.preprocessing import MultiLabelBinarizer
multilabelbinarizer = MultiLabelBinarizer()

multilabelbinarizer.fit_transform(data2['DepartmentDescription'])
multilabelbinarizer.classes_
x = pd.DataFrame(multilabelbinarizer.fit_transform(data2['DepartmentDescription']), 
                 columns = multilabelbinarizer.classes_)
data2 = pd.concat([data2,x], axis = 1)
data2 = data2.drop('DepartmentDescription', axis = 1)

# Encoding Categorical Data
x = pd.get_dummies(data2['Weekday'], drop_first = True)
data2 = pd.concat([data2,x], axis = 1)
data2 = data2.drop('Weekday', axis = 1)

x = pd.get_dummies(data2['FinelineNumber'], drop_first = True)
from sklearn.decomposition import PCA
pca = PCA(n_components = 500)
dest_small = pca.fit_transform(x)
dest_small = pd.DataFrame(dest_small)

data2 = pd.concat([data2,x], axis = 1)
data2 = data2.drop('FinelineNumber', axis = 1)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data2.iloc[:,2:] ,data2.iloc[:,0],test_size = 0.2)

# Applying Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
sum(y_pred == y_test)/18041

submit = classifier.predict_proba(X_test)
submit = pd.DataFrame(submit)
submit.columns = classifier.classes_