
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

features_list = ['poi','salary'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


# In[2]:


#view data
print len(data_dict.keys())
print data_dict.values()


# In[3]:


# remove outliers
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

# plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


# In[4]:


#fraction calculator
def fractionCreater( poi_messages, all_messages ):
    fraction = 0.
    if all_messages == 'NaN' or poi_messages == 'NaN':
        return fraction
    fraction = float(poi_messages)/float(all_messages)
    return fraction

#create new features
new_dict = {}
for name in data_dict:

    i = data_dict[name]
    from_poi_to_this_person = i["from_poi_to_this_person"]
    to_messages = i["to_messages"]
    fraction_from_poi = fractionCreater( from_poi_to_this_person, to_messages )
    i["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = i["from_this_person_to_poi"]
    from_messages = i["from_messages"]
    fraction_to_poi = fractionCreater( from_this_person_to_poi, from_messages )
    new_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    i["fraction_to_poi"] = fraction_to_poi
    
features_list = ["poi", "fraction_from_poi", "fraction_to_poi"]    
data = featureFormat(data_dict, features_list)

#plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("from poi(fraction)")
plt.ylabel("to poi(fraction)")
plt.show()


# In[5]:


from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi",
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

clf = DecisionTreeClassifier()
test_classifier(clf, data_dict, features_list, folds = 1000)

importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print ('Feature Ranking: ')
for i in range(16):
    print ("{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]]))
    


# In[6]:


#remove all with coefficent of 0

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi",
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred']

test_classifier(clf, data_dict, features_list, folds = 1000)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print ('Feature Ranking: ')
for i in range(8):
    print ("{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]]))


# In[7]:


#remove all with coefficent of 0

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi", "deferral_payments"]

test_classifier(clf, data_dict, features_list, folds = 1000)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print ('Feature Ranking: ')
for i in range(5):
    print ("{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]]))


# In[8]:


#remove deferral payments

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi"]

test_classifier(clf, data_dict, features_list, folds = 1000)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print ('Feature Ranking: ')
for i in range(4):
    print ("{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]]))


# In[9]:


#remove fraction_to_poi

features_list = ["poi", "salary", "bonus", "fraction_from_poi"]

test_classifier(clf, data_dict, features_list, folds = 1000)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print ('Feature Ranking: ')
for i in range(3):
    print ("{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]]))


# In[10]:


#precision and recall go down significantly so we go back to top 4 features

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi"]

test_classifier(clf, data_dict, features_list, folds = 1000)


# In[11]:


#check which features to use based on coefficent

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi",
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

### split data into training and testing datasets
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
test_classifier(clf, data_dict, features_list, folds = 1000)

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi"]

### split data into training and testing datasets
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
test_classifier(clf, data_dict, features_list, folds = 1000)


# In[12]:


#compare NB, Decision Tree, and SVM
from sklearn import tree

feature_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi"]

clf = GaussianNB()
print(test_classifier(clf,data_dict,feature_list))

clf = tree.DecisionTreeClassifier()
print(test_classifier(clf,data_dict,feature_list))


# In[13]:


#Tune DecisionTree because it has best precision

#Compare square root vs log 2
print('sqrt')
clf = tree.DecisionTreeClassifier(max_features = 'sqrt')
print(test_classifier(clf,data_dict,feature_list))

print('log 2')
clf = tree.DecisionTreeClassifier(max_features = 'log2')
print(test_classifier(clf,data_dict,feature_list))

print('min split of 2')
clf = tree.DecisionTreeClassifier(min_samples_split = 2)
print(test_classifier(clf,data_dict,feature_list))

print('min split of 10')
clf = tree.DecisionTreeClassifier(min_samples_split = 10)
print(test_classifier(clf,data_dict,feature_list))

print('min split of 5')
clf = tree.DecisionTreeClassifier(min_samples_split = 5)
print(test_classifier(clf,data_dict,feature_list))


# In[14]:


print ('Accuracy before tuning: ')
clf = tree.DecisionTreeClassifier()
print(test_classifier(clf,data_dict,feature_list))


print('Accuracy after tuning: ')
clf = tree.DecisionTreeClassifier(max_features = 'sqrt')
print(test_classifier(clf,data_dict,feature_list))


# In[15]:


my_dataset = data_dict
dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:




