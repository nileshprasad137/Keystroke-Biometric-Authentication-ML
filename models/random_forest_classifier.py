#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 00:43:53 2019

@author: nileshprasad137
"""

# Random Forest Classification

# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
#%%
# Importing the dataset
dataset = pd.read_csv('../dataset/DSL-StrongPasswordData.csv')

subjects = dataset["subject"].unique()

X_train = np.empty((1,31),dtype=float)
X_test = np.empty((1,31),dtype=float)
y_train = np.empty((1,),dtype=float)
y_test = np.empty((1,),dtype=float)

user_train_X = list()
user_test_X = list()
user_train_Y = list()
user_test_Y = list()

for subject in subjects: 
    genuine_user_data_X = dataset.loc[dataset.subject == subject, \
                                 "H.period":"H.Return"]
    user_X = genuine_user_data_X[:].values      
    genuine_user_data_Y = dataset.loc[dataset.subject == subject, "subject"]
    user_Y = genuine_user_data_Y[:].values    
    imposter_data = dataset.loc[dataset.subject != subject, :]   
    # Not used Currently     
    #   Uncomment Below lines when you need to set first 300 of each user as training    
#    user_train_X = genuine_user_data_X[:300].values    
#    X_train = np.append(X_train,user_train_X,axis=0)
#    user_test_X = genuine_user_data_X[300:].values
#    X_test=np.append(X_test,user_test_X,axis=0)
#    user_train_Y = genuine_user_data_Y[:300].values
#    y_train=np.append(y_train,user_train_Y,axis=0)    
#    user_test_Y = genuine_user_data_Y[300:].values
#    y_test=np.append(y_test,user_test_Y,axis=0)        
    user_train_X, user_test_X, user_train_Y, user_test_Y = train_test_split(user_X, user_Y, test_size = 0.25, random_state = 0)
    X_train = np.append(X_train,user_train_X,axis=0)
    X_test = np.append(X_test,user_test_X,axis=0)
    y_train = np.append(y_train,user_train_Y,axis=0)
    y_test = np.append(y_test,user_test_Y,axis=0)

X_train = np.delete(X_train, (0), axis=0)
X_test = np.delete(X_test, (0), axis=0)
y_train = np.delete(y_train, (0), axis=0)
y_test = np.delete(y_test, (0), axis=0)

#%%

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train.shape)
#%%
# =============================================================================
# # Fitting Random Forest Classification to the Training set
# from sklearn.ensemble import RandomForestClassifier
# random_forest_classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 5)
# random_forest_classifier.fit(X_train, y_train)
# =============================================================================
#%%

# =============================================================================
# # Predicting the Test set results
# y_pred = random_forest_classifier.predict(X_test)
# pred_vs_actual = pd.DataFrame()
# pred_vs_actual["pred"] = y_pred
# pred_vs_actual["act"] = y_test
# =============================================================================
seed = 7
np.random.seed(seed)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# Model Training 
print ("Create model ... ")
def build_model():
    model = Sequential()
    model.add(Dense(256, input_dim=31, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(160, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(90, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(51, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

print("Compile model ...")
estimator = KerasClassifier(build_fn=build_model, epochs=200, batch_size=128)
estimator.fit(X_train, y_train)

# Predictions 
print ("Predict on test data ... ")
y_pred = estimator.predict(X_test)
# =============================================================================
# #   Predicting for "Nilesh" 
# user_X_pred = dataset.loc[dataset.subject == "nilesh", \
#                                  "H.period":"H.Return"]
# user_X_values = user_X_pred[:].values   
# user_Y_pred = dataset.loc[dataset.subject == "nilesh", "subject"]
# user_Y_values = user_Y_pred[:].values
# 
# sc = StandardScaler()
# user_X_values = sc.fit_transform(user_X_values)
# 
# ans = random_forest_classifier.predict(user_X_values)
# prediction_df = pd.DataFrame()
# prediction_df["act"] = user_Y_values
# prediction_df["pred"] = ans
# =============================================================================



# =============================================================================
# ## Predicting the Test set results
# y_pred_train = random_forest_classifier.predict([X_train[0]])
# print(y_pred_train)
# pred_vs_actual_train = pd.DataFrame()
# pred_vs_actual_train["pred"] = y_pred_train
# pred_vs_actual_train["act"] = y_train[0]
# =============================================================================
#%%

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#%%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
print(classification_report(y_test,y_pred))  
print("Accuracy :: ",accuracy_score(y_test,y_pred))

# F1 Score
print("F1-Score :: ",f1_score(y_test,y_pred,average="weighted"))

# jaccard_similarity_score
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test,y_pred)
print("jaccard_similarity_score :: ",f1_score(y_test,y_pred,average="weighted"))
# =============================================================================
# 
# # Dump the trained RandomForestClassifier with Pickle
# random_forest_classifier_filename = '../saved-pickles/random_forest_classifier.pkl'
# # Open the file to save as pkl file
# random_forest_classifier_pkl = open(random_forest_classifier_filename, 'wb')
# pickle.dump(random_forest_classifier, random_forest_classifier_pkl)
# # Close the pickle instances
# random_forest_classifier_pkl.close()
# 
# =============================================================================
