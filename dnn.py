import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file1 = './sample_data.csv'
data = pd.read_csv(file1)

X = data.drop(columns=['Y'])
Y = data['Y']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)

train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y.values))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=3)

class DNN(tf.keras.models):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer = self.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
        ])
        self.fc = tf.keras.layers.Dense(1)

    def call(self, x):
        out = self.layer(x)
        return self.fc(out)

model = DNN()
model.compile(loss='mean_squared_error', optimizer='adam')

model_path = 'tf_dnn_model'
version = '1'
save_path = f'{model_path}/{version}'

model.fit(train_ds, epochs=1000, verbose=2)

pred_y = model.predict(test_X)
accuracy = np.sum(np.argmax(pred_y, axis=1) == test_Y)/len(test_Y)

tf.keras.models.save_model(model, save_path)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
#     tf.keras.layers.Dense(64,activation='relu'),
#     tf.keras.layers.Dense(64,activation='relu'),
#     tf.keras.layers.Dense(1)
#     ])
#
# model.compile(loss='mean_squared_error', optimizer='adam')
# history = model.fit(train_ds, epochs=1000, verbose=2)
# print('\n')
#
# loss = model.evaluate(test_X, test_Y, verbose=0)
# predictions = model.predict(test_X)
#
# print("Loss : ", loss)
# for i in range(5):
#     print("%d 번째 테스트 데이터의 실제값: %f" % (i, test_Y.iloc[i]))
#     print("%d 번째 테스트 데이터의 예측값: %f" % (i, predictions[i][0]))