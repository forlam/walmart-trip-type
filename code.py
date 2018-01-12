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
data = pd.read_csv('train.csv')
data.head()
data.shape

####################################################################################
####################################################################################
############################# Exploring the data ###################################
####################################################################################
####################################################################################

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

####################################################################################
####################################################################################
############################# Cleaning the dataset #################################
####################################################################################
####################################################################################

data = data.dropna(how = 'any') # Removing Null Value rows
data = data.loc[data['ScanCount'] > -1,:] # Removing rows for product return
