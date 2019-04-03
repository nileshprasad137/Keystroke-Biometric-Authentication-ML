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
   [0.06428050994873047, 0.45973706245422363, 0.5240175724029541, 0.10840272903442383, 0.06721019744873047, 0.1756129264831543, 0.04024195671081543, 0.2114391326904297, 0.2516810894012451, 0.11582112312316895, 0.4665720462799072, 0.5823931694030762, 0.05592060089111328, 0.4972536563873291, 0.5531742572784424, 0.0819089412689209, 0.44788551330566406, 0.529794454574585, 0.06311249732971191, 0.11828112602233887, 0.18139362335205078, 0.11610865592956543, 0.11819195747375488, 0.2343006134033203, 0.0671396255493164, 0.16556930541992188, 0.23270893096923828, 0.06535148620605469, 0.61002516746521, 0.6753766536712646, 0.06694960594177246]])
#prediction = model.predict(user_X)

prediction_user_rec = model.predict(user_recorded)
y_class = prediction_user_rec.argmax(axis=-1)
print(subjects[y_class[0]])
