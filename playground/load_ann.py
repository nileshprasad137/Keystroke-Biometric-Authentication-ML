# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
model = load_model('../saved-models/ann.h5')
import pandas as pd
dataset = pd.read_csv('../dataset/appended/DSL-StrongPasswordData.csv')

subjects = dataset["subject"].unique()
subjects = sorted(subjects)
# =============================================================================
# user_data_X = dataset.loc[dataset.subject == "nilesh", \
#                                  "H.period":"H.Return"]
# user_X = user_data_X[:].values 
# =============================================================================
 

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
##X_train = sc.fit_transform(user_data_X[2:])
#X_train = sc.fit_transform(user_X)
#X_test = sc.transform(user_X)

#print(subjects[])
user_recorded = np.asarray([
  [0.06636548042297363, 0.3325836658477783, 0.39894914627075195, 0.13604283332824707, 0.09629654884338379, 0.23233938217163086, 0.06706643104553223, 0.21355819702148438, 0.2806246280670166, 0.13988542556762695, 0.30579447746276855, 0.4456799030303955, 0.11603784561157227, 0.5737123489379883, 0.6897501945495605, 0.13855862617492676, 0.2009878158569336, 0.33954644203186035, 0.08198928833007812, 0.14110898971557617, 0.2230982780456543, 0.11853408813476562, 0.1030721664428711, 0.22160625457763672, 0.06769990921020508, 0.17375922203063965, 0.24145913124084473, 0.07737922668457031, 0.19434523582458496, 0.2717244625091553, 0.05939292907714844]

   ])
# =============================================================================
# prediction = model.predict(user_X)
# pred_user = prediction.argmax(axis=-1)
# for i in range(0,len(pred_user)):
#     print(subjects[pred_user[i]])
# =============================================================================

prediction_user_rec = model.predict(user_recorded)
y_class = prediction_user_rec.argmax(axis=-1)
print(subjects[y_class[0]])
