#Import data

import pandas as pd
import time

start_time_nn = time.time()

x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
x_test = pd.read_csv('x_test.csv')

#Preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Dropping columns useless for testing and training

x_test_flitered = x_test[['cfo_demod','gain_imb','iq_imb','or_off','quadr_err','ph_err','mag_err','evm']]
x_train_filtered = x_train[['cfo_demod','gain_imb','iq_imb','or_off','quadr_err','ph_err','mag_err','evm']]
y_train_filtered = y_train[['target']]

scaler = MinMaxScaler()
#scaler = StandardScaler()

x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train_filtered), columns=x_train_filtered.columns)

x_test_scaled = pd.DataFrame(scaler.fit_transform(x_test_flitered), columns=x_test_flitered.columns)

y_train_scaled=y_train_filtered-1

#Creating the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.callbacks import EarlyStopping

callbacks = EarlyStopping(monitor='val_loss', patience=10)
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(64, activation="relu"))
model.add(Dense(8, activation='softmax'))



print(model.summary())

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train_scaled, y_train_scaled, epochs=20, validation_split=0.2, batch_size=32)

end_time_nn = time.time()
elapsed_time_nn = end_time_nn - start_time_nn
print("Time NN :", elapsed_time_nn)

import numpy as np

prediction = model.predict(x_test_scaled)
print("prediction :", prediction)
print("prediction shape:", prediction.shape)
prediction_kagle=pd.DataFrame(data=np.argmax(prediction, axis=1)+1,columns=['target'])
prediction_kagle.index.name='id'
prediction_kagle.to_csv('hugo_le-fur_project_sample.csv')

#SVM 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline

start_time_svm = time.time()
columns = [1, 2, 3, 4, 5, 7, 8, 9]
x_train, x_test, y_train, y_test = train_test_split(x_train[x_train.columns[columns]], y_train[y_train.columns[1]], train_size=0.8)


df_x_test = pd.read_csv('x_test.csv')
test_x = df_x_test[df_x_test.columns[columns]]


SVM_model=make_pipeline(StandardScaler(), SVC(gamma="auto", kernel="poly", C=1000))
SVM_model.fit(x_train,y_train)

print(f"SVM Model accuracy : {SVM_model.score(x_test,y_test)}")

end_time_svm = time.time()
elapsed_time_svm = end_time_svm - start_time_svm
print("Time SVM :", elapsed_time_svm)

prediction_SVM = SVM_model.predict(test_x)
print("prediction :", prediction_SVM)
print("prediction shape:", prediction_SVM.shape)
prediction_kagle=pd.DataFrame(data=prediction_SVM,columns=['target'])
prediction_kagle.index.name='id'
prediction_kagle.to_csv('hugo_le-fur_project_sample_svm.csv')