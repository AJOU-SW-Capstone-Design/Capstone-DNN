import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(100)
tf.random.set_seed(100)

file1 = './SampleData.csv'
data = pd.read_csv(file1)

train_X = data.drop(columns=['Y'])
train_Y = data['Y']

# train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2,random_state = 42)

file2 = './TestData.csv'
data2 = pd.read_csv(file2)

test_X = data2.drop(columns=['Y'])
test_Y = data2['Y']

train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y.values))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=64)

def DNN():
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return tf_model

model = DNN()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model_path = 'tf_dnn_model'
version = '1'
save_path = f'{model_path}/{version}'

history = model.fit(train_ds, epochs=500, verbose=1)
print('\n')
loss,acc = model.evaluate(test_X, test_Y)
print("Loss : ", loss, "Acc : ", acc)

# pred_y = model.predict(test_X)
# plt.figure(figsize=(10,7))
# plt.plot(history.history['loss'], label = 'Train loss')
#
# plt.legend()
# plt.show()

# 주석해제
tf.keras.models.save_model(model, save_path)

tf_model = tf.keras.models.load_model('./tf_dnn_model/1')

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
x1_lst = [4]
x2_lst = [17]
x3_lst = [10]

data_a = pd.DataFrame((zip(x1_lst,x2_lst,x3_lst)),columns=['x1','x2','x3'])
predictions = tf_model.predict(data_a)
#
# print("Loss : ", loss)
# for i in range(5):
#     print("%d 번째 테스트 데이터의 실제값: %f" % (i, test_Y.iloc[i]))
#     print("%d 번째 테스트 데이터의 예측값: %f" % (i, predictions[i][0]))