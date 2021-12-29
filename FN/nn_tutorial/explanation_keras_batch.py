import numpy as np
from tensorflow.python import keras as K

model = K.Sequential([
    K.layers.Dense(units=4, input_shape=((2, )),# input_shape は入力の数(次元)，ベクトルは(***, )とかく
                    activation="sigmoid"),# unit は出力の数(次元)
    K.layers.Dense(units=4),
])

batch = np.random.rand(3, 2) # データ数は3なので， (3, (入力データのサイズ))と入れる

y = model.predict(batch)
print(y.shape)