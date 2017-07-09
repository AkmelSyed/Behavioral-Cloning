import pandas as pd
import numpy as np
from scipy.misc import imread
from keras.layers import Flatten, Dense, Activation, Convolution2D, normalization
from keras.models import Sequential
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

def extract_and_transform(df, col, pos):
    #Find and bring in the center picture from its residing folder
    im = imread(df[col][pos].strip())
    #Crop the image to get rid of the trees and landscape
    im = im[60:,:]
    #Resize the image
    im = cv2.resize(im, (200, 66))
    return im

def normalize(img_batch):
    #Normalize the extracted images
    norm = (np.array(img_batch)/127.5) - 1
    #Convert the image to a floating point number
    return norm.astype(np.float32)

def load_bottleneck_data(driving_log):
    file = driving_log #driving_log.csv
    #Name all the columns in the dataframe
    cols = ['Center_Image','Left_Image','Right_Image','Steering_Angle','Throttle','Break','Speed']
    df = pd.read_csv(file, names = cols)
    
    #Store all the center images
    X = []
    for i in range(len(df)):
        transformed = extract_and_transform(df, cols[0], i)
        X.append(transformed)
        
    X = normalize(X)
    #Convert the steering angles to a list and set them to y
    y = df[cols[3]].tolist()
    
    #Split the data into testing and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    return X_train, X_test, y_train, y_test


# load bottleneck data
X_train, X_test, y_train, y_test = load_bottleneck_data('C:/Users/asyed/Downloads/driving_log.csv')

#Reshape all the data for the model architecture
X_train = np.reshape(X_train, (len(X_train), 66, 200, 3))
X_test = np.reshape(X_test, (len(X_test), 66, 200, 3))
y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

input_shape = X_train.shape[1:]
#the output is a steering angle (i.e. 1 class)
nb_classes = 1


model = Sequential()
model.add(normalization.BatchNormalization())
model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', activation='relu', input_shape=input_shape))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', activation='relu'))
model.add(Flatten())

#model.add(Dropout(0.25))
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.summary()
model.compile(loss='mse', optimizer=Adam(), metrics=['mean_squared_error'])
#model.fit(X_train, y_train, nb_epoch=3, batch_size=10, validation_data=(X_test, y_test), shuffle=True)
history = model.fit_generator((X_train, y_train), 
                              samples_per_epoch=100,
                               nb_epoch=3,
                               validation_data=(X_test, y_test),
                               verbose=1) 
#Output the JSON and H5 file for trying in the driving program provided
model_json = model.to_json()
with open("C:/Users/asyed/Downloads/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("C:/Users/asyed/Downloads/model.h5")
print("Saved model to disk")