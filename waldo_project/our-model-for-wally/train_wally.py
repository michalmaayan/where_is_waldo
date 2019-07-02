import glob
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Flatten
from cnn import build_CNN

wallies = glob.glob("64/waldo/*.jpg")
non_wallies = glob.glob("64/notwaldo/*.jpg")

Y = np.concatenate([np.ones(len(wallies)), np.zeros(len(non_wallies))-1])

X = []
for name in wallies:
    X.append(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))
for name in non_wallies:
    X.append(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))
X = np.array(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
input_shape =  (3,64,64)

model = build_CNN()
model.add(Flatten())
model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save_weights("wally.h5")