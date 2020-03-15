import keras
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import os
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Activation, Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D, AveragePooling2D, Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D

parasitized = os.listdir('D:\\Back up hp 2000\\My File\\read\\Thesis\\meleria\\cell-images-for-detecting-malaria\\cell_images\\Parasitized') 
uninfected = os.listdir('D:\\Back up hp 2000\\My File\\read\\Thesis\\meleria\\cell-images-for-detecting-malaria\\cell_images\\Uninfected')
data = []
labels = []

for i in parasitized:
    try:
    
        img = cv2.imread('D:\\Back up hp 2000\\My File\\read\\Thesis\\meleria\\cell-images-for-detecting-malaria\\cell_images\\Parasitized\\'+i)
        img_array = Image.fromarray(img , 'RGB')
        resize_img = img_array.resize((64 , 64))
        rotated30 = resize_img.rotate(30)
        rotated60 = resize_img.rotate(60)
        rotated90 = resize_img.rotate(90)
        data.append(np.array(resize_img))
        data.append(np.array(rotated30))
        data.append(np.array(rotated60))
        data.append(np.array(rotated90))
        label = to_categorical(1, num_classes=2)
        labels.append(label)
        labels.append(label)
        labels.append(label)
        labels.append(label)
        
    except AttributeError:
      pass
print(len(data))

for i in uninfected:
    
    try:
        img = cv2.imread('D:\\Back up hp 2000\\My File\\read\\Thesis\\meleria\\cell-images-for-detecting-malaria\\cell_images\\Uninfected\\'+i)
        img_array = Image.fromarray(img , 'RGB')
        resize_img = img_array.resize((64 , 64))
        rotated30 = resize_img.rotate(30)
        rotated60 = resize_img.rotate(60)
        rotated90 = resize_img.rotate(90)
        data.append(np.array(resize_img))
        data.append(np.array(rotated30))
        data.append(np.array(rotated60))
        data.append(np.array(rotated90))
        label = to_categorical(0, num_classes=2)
        labels.append(label)
        labels.append(label)
        labels.append(label)
        labels.append(label)
        
    except AttributeError:
      pass
        
print(len(data))

data = np.array(data)
labels = np.array(labels)
print(data.shape , labels.shape)

print('[0.1]=Infected')
print('[1.0]=Uninfected')
plt.figure(1 , figsize = (15 , 9))
n = 0 
for i in range(20):
    n += 1 
    r = np.random.randint(0 , data.shape[0] , 1)
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.imshow(data[r[0]])
    p=labels[r[0]]
    plt.title(p) 
    plt.xticks([]) , plt.yticks([])
    
plt.show()

n = np.arange(data.shape[0])
np.random.shuffle(n)
data = data[n]
labels = labels[n]

data = data.astype(np.float32)
labels = labels.astype(np.int32)
data = data/255


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)

print(x_train.shape, x_val.shape, x_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)


#create model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),strides=(1, 1), input_shape=(64,64,3), activation='relu')) #128
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.1))
 
model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.1))
 
model.add(Conv2D(filters=128,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.1))
 
model.add(Conv2D(filters=128,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.1))
 
model.add(GlobalAveragePooling2D())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2,activation='softmax'))
 
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
train_history=model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=14, batch_size = 32)


loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val_loss'])
plt.show()

model.summary()

print(model.evaluate(x_test, y_test))
"""
from sklearn.metrics import classification_report
y_pred=model.predict(x_test)
print(classification_report(y_test,y_pred))
"""
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score

preds = model.predict(x_test)
preds = np.argmax(preds, axis = -1)
orig = np.argmax(y_test, axis=-1)

conf = confusion_matrix(orig, preds)

fig, ax = plt.subplots(figsize = (7,7))
ax.matshow(conf, cmap='Pastel1')

ax.set_ylabel('True Values')
ax.set_xlabel('Predicted Values', labelpad = 10)
ax.xaxis.set_label_position('top') 

for (i, j), z in np.ndenumerate(conf):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.show()
