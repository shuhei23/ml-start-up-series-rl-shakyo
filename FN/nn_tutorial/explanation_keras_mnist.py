import numpy as np
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from tensorflow.python import keras as K

dataset = load_digits()
image_shape = (8, 8, 1) # 1枚の8×8のグレースケール画像
num_class = 10 # 0～9の10種類(クラス)に分類する

y = dataset.target
y = K.utils.to_categorical(y, num_class)
X = dataset.data
X = np.array([data.reshape(image_shape) for data in X])#for(){X[i].reshape(image_shape)}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.33)

model = K.Sequential([
    K.layers.Conv2D(
        100, kernel_size=3, strides=1, padding="same", 
        input_shape=image_shape, activation="relu"),
    K.layers.Conv2D(
        100, kernel_size=2, strides=1, padding="same",
        activation="relu"),
    K.layers.Flatten(),
    K.layers.Dense(units=num_class, activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer="sgd")
model.fit(X_train, y_train, epochs=8)

predicts = model.predict(X_test)
predicts = np.argmax(predicts, axis=1)
actual = np.argmax(y_test, axis=1)
print(classification_report(actual, predicts))